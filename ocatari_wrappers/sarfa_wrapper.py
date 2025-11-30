"""
SARFA Saliency Wrapper for OCAtari.
Similar to GradientSaliencyWrapper but uses SARFA (Specific and Relevant Feature Attribution).
Iterates over objects instead of pixels, masking/blurring each object to compute its importance.
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
    # remove the chosen action in original output
    p = original_output[:action_index]
    p = np.append(p, original_output[action_index + 1:])
    # According to equation (2) in the paper(https://arxiv.org/abs/1912.12191v4)
    # the softmax should happen over the out put with the chosen action removed.
    # We do it like this here but we want to mention that this differs from the
    # implementation in https://github.com/nikaashpuri/sarfa-saliency/blob/master/visualize_atari/saliency.py
    p = softmax(p)

    # Do the same for the perturbed output
    new_p = perturbed_output[:action_index]
    new_p = np.append(new_p, perturbed_output[action_index + 1:])
    new_p = softmax(new_p)

    # According to the paper this should be the other way around: entropy(new_p,p)
    # (directly und er equation (2) in https://arxiv.org/pdf/1912.12191.pdf)
    # While this would make a difference, it is like this in the official implementation in
    # github.com/nikaashpuri/sarfa-saliency/blob/master/visualize_atari/saliency.py:
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


class SarfaSaliencyWrapper(MaskedBaseWrapper):
    """
    Saliency per object
    """

    def __init__(self, env, trained_model=None, use_blur=False, radius=3, use_binary_mask=False, *args, **kwargs):
        """
        Args:
            env: The environment to wrap (must have OCAtari in stack)
            trained_model: Trained PPO/DQN model for saliency computation
            use_blur: If True, use blur perturbation instead of occlusion
            radius: Radius for intensity of blur, irrelevant for blur=false
            use_binary_mask: If True, use binary masked frames instead of raw grayscale
        """
        super().__init__(env, *args, **kwargs)
        self.model = trained_model
        self.use_blur = use_blur
        self.radius = radius
        self.use_binary_mask = use_binary_mask
        self.sarfa_map = None
        # two buffers for normal and masked frames
        if not use_binary_mask:
            self.raw_buffer = deque(maxlen=self.buffer_window_size)

    def observation(self, observation):
        # use normal atari frames for sarfa computation
        # this later only takes the sarfa scores on the object boxes for the masked output
        if not self.use_binary_mask:
            raw_frame = self.unwrapped.ale.getScreenGrayscale()
            raw_frame_resized = cv2.resize(raw_frame, (84, 84), interpolation=cv2.INTER_AREA)
            self.raw_buffer.append(raw_frame_resized)
            if self.model is not None and len(self.raw_buffer) == self.buffer_window_size:
                self._compute_sarfa_map()

        # already uses masked frames
        else:
            if self.model is not None and len(self._buffer) == self.buffer_window_size:
                self._compute_sarfa_map()

        return super().observation(observation)

    def _compute_sarfa_map(self):
        # Choose frame source based on use_binary_mask flag
        if self.use_binary_mask:
            current_obs = np.asarray(self._buffer)  # Binary masked frames
        else:
            current_obs = np.asarray(self.raw_buffer)  # RAW grayscale frames

        # Get original model output
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0) / 255.0
            hidden = self.model.network(obs_tensor)
            logits = self.model.actor(hidden)
            original_output = logits.cpu().numpy()

        # Get the action being explained
        action_index = np.argmax(original_output)

        # Create 2D saliency map (84x84)
        self.sarfa_map = np.zeros((84, 84), dtype=np.float32)

        # iterate all objects
        # remove or blur them in the observation
        # compute sarfa map with each object pertubed once
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

            # skip invalid objects
            if y_max - y_min < 1 or x_max - x_min < 1:
                continue
            if y_min < 0 or x_min < 0 or y_max > 84 or x_max > 84:
                continue


            perturbed_obs = current_obs.copy()

            if self.use_blur:
                # blur entire object region, radius is only intesity
                for frame_idx in range(perturbed_obs.shape[0]):
                    object_region = perturbed_obs[frame_idx, y_min:y_max, x_min:x_max]
                    if object_region.size > 0:
                        blurred = gaussian_filter(object_region.astype(float), sigma=self.radius)
                        perturbed_obs[frame_idx, y_min:y_max, x_min:x_max] = blurred.astype(np.uint8)
            else:
                # occlude entire object
                perturbed_obs[:, y_min:y_max, x_min:x_max] = 0


            with torch.no_grad():
                perturbed_tensor = torch.FloatTensor(perturbed_obs).unsqueeze(0) / 255.0
                hidden = self.model.network(perturbed_tensor)
                logits = self.model.actor(hidden)
                perturbed_output = logits.cpu().numpy()

            # score for current object
            score = sarfa_saliency(original_output, perturbed_output, action_index)
            self.sarfa_map[y_min:y_max, x_min:x_max] = score

    def set_value(self, y_min, y_max, x_min, x_max, o):
        saliency = self._get_object_saliency(y_min, y_max, x_min, x_max)
        intensity = int(255 * np.clip(saliency, 0.0, 1.0))
        self.state[0, y_min:y_max, x_min:x_max].fill(intensity)

    def _get_object_saliency(self, y_min, y_max, x_min, x_max):
        # not none when buffer full
        if self.sarfa_map is None:
            return 0

        # transform atari frame to 84,84
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

        # normalization
        max_score = self.sarfa_map.max()
        if max_score > 0:
            saliency = mean_score / max_score
        else:
            saliency = 0
        return float(saliency)

