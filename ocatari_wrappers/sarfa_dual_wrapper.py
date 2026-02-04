"""
SARFA Dual-Channel Wrapper for OCAtari.
Provides the agent with BOTH:
1. Binary object mask (standard OCAtari representation)
2. SARFA-weighted object mask (saliency-based intensity)

The observation space is doubled: (8, 84, 84) instead of (4, 84, 84)
Channels 0-3: Binary mask frames
Channels 4-7: SARFA-weighted frames
"""

import numpy as np
import torch
import cv2
from gymnasium import spaces
from collections import deque
from scipy.special import softmax
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
from ocatari_wrappers.masked_dqn import MaskedBaseWrapper


def cross_entropy(original_output, perturbed_output, action_index):
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
    Calculate the impact of the perturbed area for the action according to SARFA formula.
    """
    original_output = np.squeeze(original_output)
    perturbed_output = np.squeeze(perturbed_output)
    dP = softmax(original_output)[action_index] - softmax(perturbed_output)[action_index]
    if dP > 0:
        K = cross_entropy(original_output, perturbed_output, action_index)
        return (2 * K * dP) / (K + dP)
    else:
        return 0


class SarfaDualWrapper(MaskedBaseWrapper):
    """
    Dual-channel wrapper: Binary + SARFA-weighted frames.
    Output shape: (8, 84, 84) - first 4 channels binary, last 4 channels SARFA-weighted.
    """

    def __init__(self, env, trained_model=None, use_blur=False, radius=3, *args, **kwargs):
        """
        Args:
            env: The environment to wrap (must have OCAtari in stack)
            trained_model: Trained PPO/DQN model for saliency computation
            use_blur: If True, use blur perturbation instead of occlusion
            radius: Radius for intensity of blur
        """
        super().__init__(env, *args, **kwargs)
        self.model = trained_model
        self.use_blur = use_blur
        self.radius = radius

        self.sarfa_map = None

        # SARFA frame buffer (separate from binary buffer in parent)
        self.sarfa_frame_buffer = deque(maxlen=self.buffer_window_size)

        # Override observation space to 8 channels
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.buffer_window_size * 2, 84, 84),  # 8 channels
            dtype=np.uint8
        )

        # SARFA intensity settings
        self.min_visible = 0
        self.use_gamma = False
        self.gamma = 0.5

    def set_model(self, model):
        """Allows injecting the agent after environment creation"""
        self.model = model

    def set_value(self, y_min, y_max, x_min, x_max, o):
        """Set binary mask value (white = 255) for objects in the state buffer."""
        self.state[0, y_min:y_max, x_min:x_max].fill(255)

    def observation(self, observation):
        # 1. Get binary observation from parent (4, 84, 84)
        binary_obs = super().observation(observation)

        # 2. Render SARFA frame directly from existing map
        sarfa_frame = self._render_sarfa_frame()
        self.sarfa_frame_buffer.append(sarfa_frame)

        # 3. Create SARFA stack (4, 84, 84)
        sarfa_obs = np.array(self.sarfa_frame_buffer)

        # 4. Combine: binary (4, 84, 84) + SARFA (4, 84, 84) = (8, 84, 84)
        combined_obs = np.concatenate([binary_obs, sarfa_obs], axis=0)

        # 5. Compute SARFA map for NEXT step (uses current combined obs)
        if self.model is not None and len(self.sarfa_frame_buffer) == self.buffer_window_size:
            self._compute_sarfa_map(combined_obs)

        return combined_obs

    def _compute_sarfa_map(self, current_obs):
        """Compute SARFA saliency map using batch GPU inference."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0) / 255.0

            if next(self.model.parameters()).is_cuda:
                obs_tensor = obs_tensor.cuda()

            hidden = self.model.network(obs_tensor)
            logits = self.model.actor(hidden)
            original_output = logits.cpu().numpy()

        action_index = np.argmax(original_output)
        self.sarfa_map = np.zeros((84, 84), dtype=np.float32)

        # Collect all valid objects and their perturbations FIRST (before GPU inference)
        perturbed_obs_list = []
        valid_objects = []

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

            perturbed_obs_list.append(perturbed_obs)
            valid_objects.append((y_min, y_max, x_min, x_max))

        # Batch GPU inference for all perturbed observations
        if perturbed_obs_list:
            batch_tensor = torch.FloatTensor(np.stack(perturbed_obs_list)) / 255.0

            if next(self.model.parameters()).is_cuda:
                batch_tensor = batch_tensor.cuda()

            hidden = self.model.network(batch_tensor)
            logits = self.model.actor(hidden)
            perturbed_outputs = logits.detach().cpu().numpy()

            # Assign saliency scores to map
            for (y_min, y_max, x_min, x_max), perturbed_output in zip(valid_objects, perturbed_outputs):
                score = sarfa_saliency(original_output, perturbed_output, action_index)
                self.sarfa_map[y_min:y_max, x_min:x_max] = score

    def _render_sarfa_frame(self):
        """Render frame directly from existing sarfa_map."""
        if self.sarfa_map is None:
            return np.zeros((84, 84), dtype=np.uint8)

        # Direct conversion: map to frame
        return (255 * np.clip(self.sarfa_map, 0.0, 1.0)).astype(np.uint8)


    def reset(self, **kwargs):
        """Reset environment and clear buffers."""
        obs, info = self.env.reset(**kwargs)

        # Clear SARFA state
        self.sarfa_frame_buffer.clear()
        self.sarfa_map = None

        # Initialize SARFA buffer with empty frames
        for _ in range(self.buffer_window_size):
            self.sarfa_frame_buffer.append(np.zeros((84, 84), dtype=np.uint8))

        # Fill binary buffer (parent's _buffer) by calling observation
        for _ in range(self.buffer_window_size):
            final_obs = self.observation(obs)

        return final_obs, info
