"""
SARFA 5-Channel Wrapper for OCAtari.
Designed for TRAINING FROM SCRATCH - no pre-trained agent required!

Provides the agent with:
1. Binary object mask (4 channels - standard OCAtari frame stack)
2. Single SARFA-weighted frame (1 channel - computed from all 4 binary frames)

The observation space is: (5, 84, 84)
Channels 0-3: Binary mask frames (frame stack)
Channel 4: SARFA saliency map (computed from channels 0-3)

Training workflow:
1. Create env with SarfaDualWrapper (model=None initially)
2. Create agent with 5-channel input
3. Call env.set_model(agent) to enable SARFA computation
4. Train normally - SARFA channel updates during training
"""

import numpy as np
import torch
from gymnasium import spaces
from scipy.special import softmax
from scipy.stats import entropy
from ocatari_wrappers.masked_dqn import BinaryMaskWrapper


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


class SarfaDualWrapper(BinaryMaskWrapper):
    """
    5-channel wrapper: 4 Binary frames + 1 SARFA saliency frame.
    Output shape: (5, 84, 84)
    - Channels 0-3: Binary mask frames (frame stack from parent)
    - Channel 4: SARFA saliency map computed from channels 0-3
    """

    def __init__(self, env, trained_model=None, compute_every_step=False, *args, **kwargs):
        """
        Args:
            env: The environment to wrap (must have OCAtari in stack)
            trained_model: Trained PPO/DQN model for saliency computation
            compute_every_step: If True, recompute SARFA every step. If False, every 4 steps.
        """
        super().__init__(env, *args, **kwargs)
        self.model = trained_model
        self.compute_every_step = compute_every_step

        self.sarfa_map = None
        self._cached_sarfa_frame = np.zeros((84, 84), dtype=np.uint8)

        # Pre-allocate output array: 4 binary + 1 SARFA = 5 channels
        self._combined_obs = np.zeros((5, 84, 84), dtype=np.uint8)

        # Override observation space to 5 channels
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(5, 84, 84),
            dtype=np.uint8
        )

        # SARFA computation interval
        self.sarfa_compute_interval = 1 if compute_every_step else 4
        self.steps_since_sarfa = 0

        # SARFA intensity settings
        self.min_visible = 40  # Minimum intensity - objects never fully invisible

    def set_model(self, model):
        """Allows injecting the agent after environment creation"""
        self.model = model

    def set_value(self, y_min, y_max, x_min, x_max, o):
        """Set binary mask value (white = 255) for objects in the state buffer."""
        self.state[0, y_min:y_max, x_min:x_max].fill(255)

    def observation(self, observation):
        # 1. Get binary observation from parent (may be less than 4 channels during buffer fill)
        binary_obs = super().observation(observation)
        n_binary = binary_obs.shape[0]

        # 2. Write binary frames into combined obs (handle partial buffer)
        self._combined_obs[:n_binary] = binary_obs

        # 3. Add cached SARFA frame as 5th channel
        self._combined_obs[4] = self._cached_sarfa_frame

        # 4. Compute SARFA map based on interval (only when buffer is full)
        if self.model is not None and n_binary == self.buffer_window_size:
            self.steps_since_sarfa += 1
            if self.steps_since_sarfa >= self.sarfa_compute_interval:
                self._compute_sarfa_map(binary_obs)
                self.steps_since_sarfa = 0

        return self._combined_obs

    def _compute_sarfa_map(self, binary_obs):
        """
        Compute SARFA saliency map from the 4 binary frames.
        Uses batched GPU inference for efficiency.
        """
        device = next(self.model.parameters()).device

        # Build 5-channel input for model (4 binary + current cached SARFA)
        model_input = np.zeros((5, 84, 84), dtype=np.uint8)
        model_input[:4] = binary_obs
        model_input[4] = self._cached_sarfa_frame

        # Get original Q-values
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(model_input).unsqueeze(0).to(device) / 255.0
            hidden = self.model.network(obs_tensor)
            logits = self.model.actor(hidden)
            original_output = logits.cpu().numpy()

        action_index = np.argmax(original_output)
        self.sarfa_map = np.zeros((84, 84), dtype=np.float32)

        # Collect valid object bounding boxes
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

        # Batched inference: create perturbed versions for all objects
        n_objects = len(valid_objects)
        batch_tensor = torch.FloatTensor(model_input).unsqueeze(0).expand(n_objects, -1, -1, -1).clone().to(device) / 255.0

        # Apply occlusion: set object region to 0 in ALL 5 channels for each object
        for i, (y_min, y_max, x_min, x_max) in enumerate(valid_objects):
            batch_tensor[i, :, y_min:y_max, x_min:x_max] = 0

        # Single batched forward pass
        with torch.no_grad():
            hidden = self.model.network(batch_tensor)
            logits = self.model.actor(hidden)
            perturbed_outputs = logits.cpu().numpy()

        # Assign saliency scores to map
        object_mask = np.zeros((84, 84), dtype=bool)
        for (y_min, y_max, x_min, x_max), perturbed_output in zip(valid_objects, perturbed_outputs):
            score = sarfa_saliency(original_output, perturbed_output, action_index)
            self.sarfa_map[y_min:y_max, x_min:x_max] = score
            object_mask[y_min:y_max, x_min:x_max] = True

        # Normalize saliency map
        max_score = self.sarfa_map.max()
        if max_score > 0:
            self.sarfa_map /= max_score

        # Cache the rendered frame with min_visible floor
        frame = (self.sarfa_map * 255).astype(np.uint8)
        frame[object_mask] = np.maximum(frame[object_mask], self.min_visible)
        self._cached_sarfa_frame[:] = frame

    def reset(self, **kwargs):
        """Reset environment and clear buffers."""
        obs, info = self.env.reset(**kwargs)

        # Clear SARFA state
        self.sarfa_map = None
        self._cached_sarfa_frame.fill(0)
        self.steps_since_sarfa = 0

        # Fill binary buffer by calling observation multiple times
        for _ in range(self.buffer_window_size):
            final_obs = self.observation(obs)

        return final_obs, info
