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
from gymnasium import spaces
from collections import deque
from scipy.special import softmax
from scipy.stats import entropy
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
        self._cached_sarfa_frame = np.zeros((84, 84), dtype=np.uint8)

        # SARFA frame buffer (separate from binary buffer in parent)
        self.sarfa_frame_buffer = deque(maxlen=self.buffer_window_size)

        # Pre-allocate output array to avoid concatenation overhead
        self._combined_obs = np.zeros((self.buffer_window_size * 2, 84, 84), dtype=np.uint8)

        # Override observation space to 8 channels
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.buffer_window_size * 2, 84, 84),  # 8 channels
            dtype=np.uint8
        )

        # SARFA computation interval - only recompute every N steps
        self.sarfa_compute_interval = 4  # Recompute every 4 steps (same as frame stack)
        self.steps_since_sarfa = 0

        # SARFA intensity settings
        self.min_visible = 40  # Minimum intensity - objects never fully invisible
        self.use_gamma = False
        self.gamma = 0.5

    def set_model(self, model):
        """Allows injecting the agent after environment creation"""
        self.model = model

    def set_value(self, y_min, y_max, x_min, x_max, o):
        """Set binary mask value (white = 255) for objects in the state buffer."""
        self.state[0, y_min:y_max, x_min:x_max].fill(255)

    def observation(self, observation):
        # 1. Get binary observation from parent
        binary_obs = super().observation(observation)
        n_binary = binary_obs.shape[0]

        # 2. Add current cached SARFA frame to buffer
        self.sarfa_frame_buffer.append(self._cached_sarfa_frame.copy())

        # 3. Write into pre-allocated array
        self._combined_obs[:n_binary] = binary_obs
        for i, frame in enumerate(self.sarfa_frame_buffer):
            self._combined_obs[4 + i] = frame

        # 4. Compute SARFA map every N steps (updates _cached_sarfa_frame for NEXT observation)
        if (self.model is not None and
            n_binary == self.buffer_window_size and
            len(self.sarfa_frame_buffer) == self.buffer_window_size):
            self.steps_since_sarfa += 1
            if self.steps_since_sarfa >= self.sarfa_compute_interval:
                self._compute_sarfa_map(self._combined_obs)
                self.steps_since_sarfa = 0

        return self._combined_obs

    def _compute_sarfa_map(self, current_obs):
        """Compute SARFA saliency map using batch GPU inference."""
        device = next(self.model.parameters()).device

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0).to(device) / 255.0
            hidden = self.model.network(obs_tensor)
            logits = self.model.actor(hidden)
            original_output = logits.cpu().numpy()

        action_index = np.argmax(original_output)
        self.sarfa_map = np.zeros((84, 84), dtype=np.float32)

        # Collect valid object bounding boxes first
        valid_objects = []
        for obj in self.env.objects:
            if obj is None or obj.category == "NoObject":
                continue

            x, y, w, h = obj.xywh
            y_min = int(y * 84 / 210)
            y_max = int((y + h) * 84 / 210)
            x_min = int(x * 84 / 160)
            x_max = int((x + w) * 84 / 160)

            if y_max - y_min < 1 or x_max - x_min < 1:
                continue
            if y_min < 0 or x_min < 0 or y_max > 84 or x_max > 84:
                continue

            valid_objects.append((y_min, y_max, x_min, x_max))

        if not valid_objects:
            self._cached_sarfa_frame.fill(0)
            return

        # Pre-allocate batch tensor directly on GPU
        n_objects = len(valid_objects)
        batch_tensor = torch.FloatTensor(current_obs).unsqueeze(0).expand(n_objects, -1, -1, -1).clone().to(device) / 255.0

        # Apply occlusion directly on tensor
        for i, (y_min, y_max, x_min, x_max) in enumerate(valid_objects):
            batch_tensor[i, :, y_min:y_max, x_min:x_max] = 0

        # Single batched forward pass
        with torch.no_grad():
            hidden = self.model.network(batch_tensor)
            logits = self.model.actor(hidden)
            perturbed_outputs = logits.cpu().numpy()

        # Assign saliency scores to map AND track object regions
        object_mask = np.zeros((84, 84), dtype=bool)
        for (y_min, y_max, x_min, x_max), perturbed_output in zip(valid_objects, perturbed_outputs):
            score = sarfa_saliency(original_output, perturbed_output, action_index)
            self.sarfa_map[y_min:y_max, x_min:x_max] = score
            object_mask[y_min:y_max, x_min:x_max] = True

        # Normalize saliency map
        max_score = self.sarfa_map.max()
        if max_score > 0:
            self.sarfa_map /= max_score

        # Cache the rendered frame with min_visible floor for ALL objects
        frame = (self.sarfa_map * 255).astype(np.uint8)
        # Apply minimum visibility - ALL objects should never be completely invisible
        frame[object_mask] = np.maximum(frame[object_mask], self.min_visible)
        self._cached_sarfa_frame[:] = frame


    def reset(self, **kwargs):
        """Reset environment and clear buffers."""
        obs, info = self.env.reset(**kwargs)

        # Clear SARFA state
        self.sarfa_frame_buffer.clear()
        self.sarfa_map = None
        self._cached_sarfa_frame.fill(0)
        self.steps_since_sarfa = 0

        # Initialize SARFA buffer with empty frames
        for _ in range(self.buffer_window_size):
            self.sarfa_frame_buffer.append(np.zeros((84, 84), dtype=np.uint8))

        # Fill binary buffer (parent's _buffer) by calling observation
        for _ in range(self.buffer_window_size):
            final_obs = self.observation(obs)

        return final_obs, info
