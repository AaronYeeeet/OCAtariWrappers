"""
SARFA Saliency Wrapper for OCAtari.
Similar to GradientSaliencyWrapper but uses SARFA (Specific and Relevant Feature Attribution).
Iterates over objects instead of pixels, masking/blurring each object to compute its importance.
"""

import numpy as np
import torch
import cv2
from collections import deque
from ocatari_wrappers.masked_dqn import BinaryMaskWrapper
from ocatari_wrappers.sarfa_common import sarfa_saliency


class SarfaSaliencyWrapper(BinaryMaskWrapper):
    """
    Saliency per object
    """

    def __init__(self, env, trained_model=None, use_binary_mask=True, use_fade_in=True, fade_in_steps = 50000, *args, **kwargs):
        """
        Args:
            env: The environment to wrap (must have OCAtari in stack)
            trained_model: Trained PPO/DQN model for saliency computation
            use_binary_mask: If True, use binary masked frames instead of raw grayscale
        """
        super().__init__(env, *args, **kwargs)
        self.model = trained_model
        self.use_binary_mask = use_binary_mask

        # buffer for raw grayscale frames (only needed when not using binary mask)
        if not use_binary_mask:
            self.raw_buffer = deque(maxlen=self.buffer_window_size)

        self.use_fade_in = use_fade_in  # FADE IN
        self.fade_in_steps = fade_in_steps # 10 is number environments in cleanRL. Divide X by num envs if you want X global step fade
        self.sarfa_step_counter = 0
        self.min_visible = 0  # Minimum intensity after fade-in (0-255), objects never fully invisible
        # 0 for normal without minimum visibility

        self.use_gamma = False  # use power function instead of min_visible
        self.gamma = 0.5  # gamma < 1 hebt niedrige Werte an, gamma > 1 senkt sie (0.5 = Quadratwurzel)

    def set_model(self, model):
        """Allows injecting the agent after environment creation"""
        self.model = model

    def observation(self, observation):
        # 1. Raw-Buffer für die spätere SARFA-Berechnung füllen
        if not self.use_binary_mask:
            raw_frame = self.unwrapped.ale.getScreenGrayscale()
            raw_frame_resized = cv2.resize(raw_frame, (84, 84), interpolation=cv2.INTER_AREA)
            self.raw_buffer.append(raw_frame_resized)

        # 2. Prepare state array (like MaskedBaseWrapper.observation())
        self.state = np.zeros(self.working_shape, dtype=np.uint8)

        # 3. Compute SARFA and fill state directly (no double iteration!)
        if not self.use_binary_mask:
            if self.model is not None and len(self.raw_buffer) == self.buffer_window_size:
                self._compute_sarfa_map()
            else:
                self._fill_state_white()  # Fallback: white boxes until buffer is full
        else:
            if self.model is not None and len(self._buffer) == self.buffer_window_size:
                self._compute_sarfa_map()
            else:
                self._fill_state_white()  # Fallback: white boxes until buffer is full

        if self.use_fade_in:
            self.sarfa_step_counter += 1

        # Skip parent's observation() - we already filled self.state!
        return self.create_obs(self.state)

    def _fill_state_white(self):
        """Fallback: Fill all objects with white (255) until SARFA is ready."""
        for obj in self.env.objects:
            if obj is None or obj.category == "NoObject":
                continue
            x, y, w, h = obj.xywh
            y_min, y_max, x_min, x_max = self.calc_limits(x, y, x + w, y + h)
            if y_max > y_min and x_max > x_min:
                self.state[0, y_min:y_max, x_min:x_max].fill(255)

    def _compute_sarfa_map(self):
        """Compute SARFA scores and fill self.state directly - single iteration over objects."""
        # Choose frame source based on use_binary_mask flag
        if self.use_binary_mask:
            current_obs = np.asarray(self._buffer)  # Binary masked frames
        else:
            current_obs = np.asarray(self.raw_buffer)  # RAW grayscale frames

        # Get original model output
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0) / 255.0

            # CUDA Check: Ensure input is on same device as model
            if next(self.model.parameters()).is_cuda:
                obs_tensor = obs_tensor.cuda()

            hidden = self.model.network(obs_tensor)
            logits = self.model.actor(hidden)
            original_output = logits.cpu().numpy()

        # Get the action being explained
        action_index = np.argmax(original_output)

        # First pass: collect all SARFA scores for normalization
        object_scores = []
        object_coords = []

        for obj in self.env.objects:
            if obj is None or obj.category == "NoObject":
                continue

            x, y, w, h = obj.xywh
            height_orig, width_orig = 210, 160
            height_grad, width_grad = 84, 84
            y_min_obs = int(y * height_grad / height_orig)
            y_max_obs = int((y + h) * height_grad / height_orig)
            x_min_obs = int(x * width_grad / width_orig)
            x_max_obs = int((x + w) * width_grad / width_orig)

            # skip invalid objects
            if y_max_obs - y_min_obs < 1 or x_max_obs - x_min_obs < 1:
                continue
            if y_min_obs < 0 or x_min_obs < 0 or y_max_obs > 84 or x_max_obs > 84:
                continue

            perturbed_obs = current_obs.copy()

            # occlude entire object
            perturbed_obs[:, y_min_obs:y_max_obs, x_min_obs:x_max_obs] = 0

            with torch.no_grad():
                perturbed_tensor = torch.FloatTensor(perturbed_obs).unsqueeze(0) / 255.0

                # CUDA Check for perturbed input
                if next(self.model.parameters()).is_cuda:
                    perturbed_tensor = perturbed_tensor.cuda()

                hidden = self.model.network(perturbed_tensor)
                logits = self.model.actor(hidden)
                perturbed_output = logits.cpu().numpy()

            # score for current object
            score = sarfa_saliency(original_output, perturbed_output, action_index)

            # Get state coordinates (using parent's calc_limits)
            y_min_state, y_max_state, x_min_state, x_max_state = self.calc_limits(x, y, x + w, y + h)

            object_scores.append(score)
            object_coords.append((y_min_state, y_max_state, x_min_state, x_max_state))

        # Normalize and fill state
        max_score = max(object_scores) if object_scores else 0

        for score, (y_min, y_max, x_min, x_max) in zip(object_scores, object_coords):
            # Normalize
            if max_score > 0:
                saliency = score / max_score
            else:
                saliency = 0

            # Apply gamma transformation if enabled
            if self.use_gamma:
                saliency = np.power(np.clip(saliency, 0.0, 1.0), self.gamma)

            intensity = int(255 * np.clip(saliency, 0.0, 1.0))

            # Calculate final intensity with fade-in or min_visible
            if self.use_fade_in and self.sarfa_step_counter < self.fade_in_steps:
                alpha = min(1.0, self.sarfa_step_counter / self.fade_in_steps)
                final_intensity = int((1.0 - alpha) * 255 + alpha * intensity)
            elif self.use_gamma:
                final_intensity = intensity
            else:
                final_intensity = max(self.min_visible, intensity)

            # Fill state directly
            if y_max > y_min and x_max > x_min:
                self.state[0, y_min:y_max, x_min:x_max].fill(final_intensity)

