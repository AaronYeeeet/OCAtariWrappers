"""
SARFA Dropout Wrapper for OCAtari.
Removes the N most salient objects from the observation before passing to agent.
Uses SARFA saliency scores to determine importance.
"""

import numpy as np
import torch
import cv2
from collections import deque
from scipy.special import softmax
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
from ocatari_wrappers.masked_dqn import MaskedBaseWrapper


def cross_entropy(original_output, perturbed_output, action_index):
    """Compute cross-entropy for SARFA calculation."""
    p = original_output[:action_index]
    p = np.append(p, original_output[action_index + 1:])
    p = softmax(p)

    new_p = perturbed_output[:action_index]
    new_p = np.append(new_p, perturbed_output[action_index + 1:])
    new_p = softmax(new_p)

    KL = entropy(p, new_p)
    K = 1. / (1. + KL)
    return K


def sarfa_saliency(original_output, perturbed_output, action_index):
    """
    Calculate the impact of the perturbed area in *perturbed_output* for the action *action_index*
    according to the SARFA formula.
    """
    original_output = np.squeeze(original_output)
    perturbed_output = np.squeeze(perturbed_output)
    dP = softmax(original_output)[action_index] - softmax(perturbed_output)[action_index]
    if dP > 0:
        K = cross_entropy(original_output, perturbed_output, action_index)
        return (2 * K * dP) / (K + dP)
    else:
        return 0


class SarfaDropoutWrapper(MaskedBaseWrapper):
    """
    Wrapper that removes the N most salient objects from the observation.
    The remaining objects are shown with normalized saliency intensities.

    Uses SARFA saliency to determine object importance and drops the most
    important objects (highest saliency scores) from the observation.
    """

    def __init__(self, env, trained_model=None, n_dropout=1, use_blur=False,
                 radius=3, use_binary_mask=False, normalize_remaining=True, *args, **kwargs):
        """
        Args:
            env: The environment to wrap (must have OCAtari in stack)
            trained_model: Trained PPO/DQN model for saliency computation
            n_dropout: Number of most salient objects to remove from observation
            use_blur: If True, use blur perturbation instead of occlusion for saliency computation
            radius: Radius for intensity of blur (irrelevant if use_blur=False)
            use_binary_mask: If True, use binary masked frames instead of raw grayscale
            normalize_remaining: If True, normalize intensities of remaining objects after dropout
        """
        super().__init__(env, *args, **kwargs)
        self.model = trained_model
        self.n_dropout = n_dropout
        self.use_blur = use_blur
        self.radius = radius
        self.use_binary_mask = use_binary_mask
        self.normalize_remaining = normalize_remaining

        # Saliency tracking
        self.sarfa_map = None
        self.object_saliencies = {}  # {object_id: saliency_score}
        self.dropped_objects = set()  # Set of object categories/ids to drop

        # Buffer for raw frames (when not using binary mask)
        if not use_binary_mask:
            self.raw_buffer = deque(maxlen=self.buffer_window_size)

    def observation(self, observation):
        # Collect raw frames for SARFA computation
        if not self.use_binary_mask:
            raw_frame = self.unwrapped.ale.getScreenGrayscale()
            raw_frame_resized = cv2.resize(raw_frame, (84, 84), interpolation=cv2.INTER_AREA)
            self.raw_buffer.append(raw_frame_resized)
            if self.model is not None and len(self.raw_buffer) == self.buffer_window_size:
                self._compute_sarfa_and_dropout()
        else:
            if self.model is not None and len(self._buffer) == self.buffer_window_size:
                self._compute_sarfa_and_dropout()

        return super().observation(observation)

    def _compute_sarfa_and_dropout(self):
        """Compute SARFA saliency for all objects and determine which to drop."""
        if self.use_binary_mask:
            current_obs = (np.asarray(self._buffer) * 255).astype(np.uint8)
        else:
            current_obs = np.asarray(self.raw_buffer)

        # Get original model output
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0) / 255.0
            hidden = self.model.network(obs_tensor)
            logits = self.model.actor(hidden)
            original_output = logits.cpu().numpy()

        action_index = np.argmax(original_output)

        # Reset tracking
        self.sarfa_map = np.zeros((84, 84), dtype=np.float32)
        self.object_saliencies = {}

        # Compute saliency for each object
        for obj in self.env.objects:
            if obj is None or obj.category == "NoObject":
                continue

            x, y, w, h = obj.xywh
            height_orig, width_orig = 210, 160
            height_grad, width_grad = 84, 84
            y_min = int(y * height_grad / height_orig)
            y_max = int((y + h) * height_grad / height_orig)
            x_min = int(x * width_grad / width_orig)
            x_max = int((x + w) * width_grad / width_orig)

            # Skip invalid objects
            if y_max - y_min < 1 or x_max - x_min < 1:
                continue
            if y_min < 0 or x_min < 0 or y_max > 84 or x_max > 84:
                continue

            perturbed_obs = current_obs.copy()

            if self.use_blur:
                for frame_idx in range(perturbed_obs.shape[0]):
                    object_region = perturbed_obs[frame_idx, y_min:y_max, x_min:x_max]
                    if object_region.size > 0:
                        blurred = gaussian_filter(object_region.astype(float), sigma=self.radius)
                        perturbed_obs[frame_idx, y_min:y_max, x_min:x_max] = blurred.astype(np.uint8)
            else:
                perturbed_obs[:, y_min:y_max, x_min:x_max] = 0

            with torch.no_grad():
                perturbed_tensor = torch.FloatTensor(perturbed_obs).unsqueeze(0) / 255.0
                hidden = self.model.network(perturbed_tensor)
                logits = self.model.actor(hidden)
                perturbed_output = logits.cpu().numpy()

            score = sarfa_saliency(original_output, perturbed_output, action_index)
            self.sarfa_map[y_min:y_max, x_min:x_max] = score

            # Store saliency per object (use id to distinguish same-category objects)
            obj_key = (obj.category, id(obj))
            self.object_saliencies[obj_key] = score

        # Determine which objects to drop (top N by saliency)
        self._update_dropout_set()

    def _update_dropout_set(self):
        """Update the set of objects to drop based on saliency scores."""
        if not self.object_saliencies:
            self.dropped_objects = set()
            return

        # Sort objects by saliency (highest first)
        sorted_objects = sorted(
            self.object_saliencies.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Take top N objects to drop
        self.dropped_objects = set(
            obj_key for obj_key, _ in sorted_objects[:self.n_dropout]
        )

    def set_value(self, y_min, y_max, x_min, x_max, o):
        """
        Set the value for an object in the observation.
        Dropped objects get intensity 0 (removed).
        Remaining objects get normalized saliency intensity.
        """
        obj_key = (o.category, id(o))

        # Check if this object should be dropped
        if obj_key in self.dropped_objects:
            # Object is dropped - don't draw it (leave as 0/black)
            return

        # Get saliency for remaining objects
        saliency = self._get_object_saliency(y_min, y_max, x_min, x_max, o)

        if self.normalize_remaining:
            # Normalize among remaining objects only
            intensity = int(255 * np.clip(saliency, 0.0, 1.0))
        else:
            # Use raw saliency (may not use full intensity range)
            intensity = int(255 * np.clip(saliency, 0.0, 1.0))

        self.state[0, y_min:y_max, x_min:x_max].fill(intensity)

    def _get_object_saliency(self, y_min, y_max, x_min, x_max, obj=None):
        """Get normalized saliency for an object."""
        if self.sarfa_map is None:
            return 0

        # Transform coordinates to sarfa_map scale
        height_orig, width_orig = self.state.shape[1], self.state.shape[2]
        height_grad, width_grad = self.sarfa_map.shape
        y_min_scaled = int(y_min * height_grad / height_orig)
        y_max_scaled = int(y_max * height_grad / height_orig)
        x_min_scaled = int(x_min * width_grad / width_orig)
        x_max_scaled = int(x_max * width_grad / width_orig)

        object_scores = self.sarfa_map[y_min_scaled:y_max_scaled,
                                       x_min_scaled:x_max_scaled]

        if object_scores.size > 0:
            mean_score = object_scores.mean()
        else:
            mean_score = 0

        if self.normalize_remaining:
            # Normalize only among remaining (non-dropped) objects
            remaining_saliencies = [
                score for obj_key, score in self.object_saliencies.items()
                if obj_key not in self.dropped_objects
            ]
            if remaining_saliencies:
                max_remaining = max(remaining_saliencies)
                if max_remaining > 0:
                    return float(mean_score / max_remaining)
            return 0.0
        else:
            # Normalize against all objects
            max_score = self.sarfa_map.max()
            if max_score > 0:
                return float(mean_score / max_score)
            return 0.0

    def get_dropped_object_info(self):
        """
        Get information about the currently dropped objects.

        Returns:
            list: List of tuples (category, saliency_score) for dropped objects
        """
        dropped_info = []
        for obj_key in self.dropped_objects:
            category = obj_key[0]
            score = self.object_saliencies.get(obj_key, 0)
            dropped_info.append((category, score))
        return dropped_info

    def get_remaining_object_info(self):
        """
        Get information about the remaining (non-dropped) objects.

        Returns:
            list: List of tuples (category, saliency_score) for remaining objects
        """
        remaining_info = []
        for obj_key, score in self.object_saliencies.items():
            if obj_key not in self.dropped_objects:
                category = obj_key[0]
                remaining_info.append((category, score))
        return remaining_info

