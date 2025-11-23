"""
SARFA (Saliency-based Randomized Feature Ablation) Wrapper for OCAtari environments.
Generates saliency maps by randomly perturbing regions and measuring action value changes.

Based on SARFA implementation from:
https://github.com/nikaashpuri/sarfa-saliency/blob/master/visualize_atari/saliency.py
and https://arxiv.org/pdf/1912.12191.pdf
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import softmax
from scipy.stats import entropy
from PIL import Image
import torch

from ocatari_wrappers.masked_dqn import MaskedBaseWrapper


def cross_entropy(original_output, perturbed_output, action_index):
    """
    Calculate cross entropy between original and perturbed outputs.

    Args:
        original_output: Q-values from original state
        perturbed_output: Q-values from perturbed state
        action_index: Index of the action to analyze

    Returns:
        K: Cross entropy metric
    """
    # Remove the chosen action in original output
    p = original_output[:action_index]
    p = np.append(p, original_output[action_index + 1:])
    # According to equation (2) in the paper(https://arxiv.org/abs/1912.12191v4)
    # the softmax should happen over the output with the chosen action removed.
    p = softmax(p)

    # Do the same for the perturbed output
    new_p = perturbed_output[:action_index]
    new_p = np.append(new_p, perturbed_output[action_index + 1:])
    new_p = softmax(new_p)

    # According to the paper this should be the other way around: entropy(new_p,p)
    # (directly under equation (2) in https://arxiv.org/pdf/1912.12191.pdf)
    # While this would make a difference, it is like this in the official implementation
    KL = entropy(p, new_p)
    K = 1. / (1. + KL)

    return K


def sarfa_saliency(original_output, perturbed_output, action_index):
    """
    Calculate the impact of the perturbed area according to the SARFA formula.

    Args:
        original_output: Q-values from original state
        perturbed_output: Q-values from perturbed state
        action_index: Index of the action to analyze

    Returns:
        SARFA saliency score for this perturbation
    """
    original_output = np.squeeze(original_output)
    perturbed_output = np.squeeze(perturbed_output)
    dP = softmax(original_output)[action_index] - softmax(perturbed_output)[action_index]
    if dP > 0:
        K = cross_entropy(original_output, perturbed_output, action_index)
        return (2 * K * dP) / (K + dP)
    else:
        return 0


class SARFAWrapper(MaskedBaseWrapper):
    """
    Wrapper that adds SARFA saliency map generation capabilities to OCAtari environments.

    SARFA (Saliency-based Randomized Feature Ablation) generates saliency maps by
    randomly perturbing circular regions and measuring the impact on action probabilities.

    Args:
        env: OCAtari environment to wrap
        trained_model: Trained agent model for Q-value predictions (can be PyTorch or Keras)
        r: Radius of perturbation region (default: 5)
        blur: Whether to use Gaussian blur for perturbation instead of black occlusion (default: True)
        density: Sampling density - affects resolution of saliency map (default: 5)
        *args, **kwargs: Additional arguments for MaskedBaseWrapper
    """

    def __init__(self, env, trained_model=None, r: int = 5, blur: bool = True, density: int = 5, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.model = trained_model
        self.r = r
        self.blur = blur
        self.density = density
        self.last_saliency_map = None

        # Detect if model is PyTorch or Keras
        self.is_pytorch = trained_model is not None and hasattr(trained_model, 'parameters')

        if self.is_pytorch and trained_model is not None:
            self.device = next(trained_model.parameters()).device
        else:
            self.device = None

    def _get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """
        Get Q-values (or action logits) from the model for given observation.

        Args:
            obs: Observation to get Q-values for

        Returns:
            Q-values/action logits as numpy array
        """
        if obs.ndim == 3:
            obs = np.expand_dims(obs, axis=0)

        if self.is_pytorch:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.device)

                # Check if this is a PPO-style model (has actor and network)
                if hasattr(self.model, 'actor') and hasattr(self.model, 'network'):
                    # PPO model: get action logits from actor
                    hidden = self.model.network(obs_tensor / 255.0)
                    logits = self.model.actor(hidden).cpu().numpy()
                    return logits[0]
                elif hasattr(self.model, 'get_value'):
                    # DQN-style model: try get_value
                    q_values = self.model.get_value(obs_tensor).cpu().numpy()
                    return q_values[0] if q_values.ndim > 1 else q_values
                else:
                    # Generic: call model directly
                    output = self.model(obs_tensor).cpu().numpy()
                    return output[0]
        else:
            # Keras model
            return self.model.predict(obs, verbose=0)[0]

    def _get_occlusion_mask(self, center, size, radius):
        """
        Creates a circular mask to occlude the image with black color.

        Args:
            center: Center position of the mask (y, x)
            size: Size of the mask (height, width)
            radius: Radius of the circular mask

        Returns:
            Binary mask with 1s in circular region
        """
        y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
        # Distance to center (calculated with pythagoras) has to be lower than or equal to radius
        keep = x * x + y * y <= radius * radius
        mask = np.zeros(size)
        mask[keep] = 1  # Select a circle of pixels
        return mask

    def _get_blur_mask(self, center, size, radius):
        """
        Creates a Gaussian blurred mask for perturbation.

        Args:
            center: Center position of mask (y, x)
            size: Size of the mask (height, width)
            radius: Radius of the blurring

        Returns:
            Gaussian blurred mask
        """
        y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
        keep = x * x + y * y <= 1
        mask = np.zeros(size)
        mask[keep] = 1  # Select a circle of pixels
        mask = gaussian_filter(mask, sigma=radius)  # Blur the circle of pixels
        return mask / mask.max()

    @staticmethod
    def _occlude(image, mask):
        """Apply black occlusion using mask."""
        return image * (1 - mask)

    @staticmethod
    def _occlude_blur(image, mask):
        """Apply Gaussian blur occlusion using mask."""
        return image * (1 - mask) + gaussian_filter(image, sigma=3) * mask

    def generate_saliency_map(self, obs: np.ndarray, action: int = None) -> np.ndarray:
        """
        Generate SARFA saliency map for given observation and action.

        Args:
            obs: Current observation/state (can be stacked frames)
            action: Action to explain (if None, uses best action)

        Returns:
            Saliency map of shape (84, 84)
        """
        if self.model is None:
            raise ValueError("No model provided for SARFA saliency computation")

        # Handle different input shapes
        if obs.ndim == 4:
            stacked_frames = np.squeeze(obs, axis=0)
        else:
            stacked_frames = obs

        # OCAtari uses channel-first (4, 84, 84), but SARFA needs channel-last (84, 84, 4)
        # Convert for SARFA processing
        if stacked_frames.ndim == 3 and stacked_frames.shape[0] in [1, 3, 4]:
            # Channel-first detected, transpose to channel-last
            stacked_frames_for_sarfa = np.transpose(stacked_frames, (1, 2, 0))
            stacked_frames_for_model = stacked_frames  # Keep original for model
        else:
            stacked_frames_for_sarfa = stacked_frames
            stacked_frames_for_model = stacked_frames

        # Get original Q-values (using channel-first format for model)
        original_output = self._get_q_values(stacked_frames_for_model)

        # Determine action to explain
        if action is None:
            action_index = np.argmax(original_output)
        else:
            action_index = action

        # Use channel-last dimensions for SARFA processing
        x = stacked_frames_for_sarfa.shape[0]  # Height (84)
        y = stacked_frames_for_sarfa.shape[1]  # Width (84)

        # Density d: if d==1, get a score for every pixel
        # if d==2, then every other pixel (25% of total pixels)
        d = self.density

        # Initialize scores array
        scores = np.zeros((int((x-1) / d) + 1, int((y-1) / d) + 1))

        # Iterate over all positions with density d
        for i in range(0, x, d):
            for j in range(0, y, d):
                # Create mask for this position (2D spatial mask)
                if self.blur:
                    mask = self._get_blur_mask(center=[i, j], size=[x, y], radius=self.r)
                else:
                    mask = self._get_occlusion_mask(center=[i, j], size=[x, y], radius=self.r)

                # Stack mask for all frames (channel-last format)
                stacked_mask = np.zeros(shape=stacked_frames_for_sarfa.shape)
                for idx in range(stacked_frames_for_sarfa.shape[2]):  # Iterate over channels
                    stacked_mask[:, :, idx] = mask

                # Apply perturbation (in channel-last format)
                if self.blur:
                    perturbed_frames_cl = self._occlude_blur(stacked_frames_for_sarfa, stacked_mask)
                else:
                    perturbed_frames_cl = self._occlude(stacked_frames_for_sarfa, stacked_mask)

                # Convert back to channel-first for model prediction
                perturbed_frames_cf = np.transpose(perturbed_frames_cl, (2, 0, 1))

                # Get Q-values for perturbed state
                perturbed_output = self._get_q_values(perturbed_frames_cf)

                # Calculate SARFA saliency score
                scores[int(i / d), int(j / d)] = sarfa_saliency(
                    original_output, perturbed_output, action_index
                )

        # Resize to original dimensions and normalize
        pmax = scores.max()
        # Use Image.Resampling.BILINEAR for newer PIL versions, fallback to Image.BILINEAR
        try:
            from PIL.Image import Resampling
            resample_method = Resampling.BILINEAR
        except ImportError:
            resample_method = Image.BILINEAR
        scores = Image.fromarray(scores).resize(size=[x, y], resample=resample_method)
        scores = np.array(scores)
        if scores.max() > 0:
            scores = pmax * scores / scores.max()

        self.last_saliency_map = scores
        return scores

    def set_value(self, y_min, y_max, x_min, x_max, o):
        """
        Set value in masked state.

        If no saliency map is available (normal gameplay), behaves like BinaryMaskWrapper (white).
        If saliency map exists, uses the average saliency for this object.

        This is called by the observation() method for each object.
        """
        # If no saliency map available, use white (255) like BinaryMaskWrapper
        saliency = self._get_object_saliency(y_min, y_max, x_min, x_max)
        intensity = int(255 * np.clip(saliency, 0.0, 1.0)) if saliency > 0 else 255
        self.state[0, y_min:y_max, x_min:x_max].fill(intensity)

    def _get_object_saliency(self, y_min, y_max, x_min, x_max):
        """
        Get average SARFA saliency for an object's bounding box.

        Args:
            y_min, y_max, x_min, x_max: Bounding box coordinates

        Returns:
            Normalized saliency value for the object
        """
        # Skip if no saliency map available
        if self.last_saliency_map is None:
            return 0

        # Adjust coordinates to saliency map dimensions
        height_orig, width_orig = self.state.shape[1], self.state.shape[2]
        height_sal, width_sal = self.last_saliency_map.shape

        y_min_scaled = int(y_min * height_sal / height_orig)
        y_max_scaled = int(y_max * height_sal / height_orig)
        x_min_scaled = int(x_min * width_sal / width_orig)
        x_max_scaled = int(x_max * width_sal / width_orig)

        # Get saliency values for this object
        object_saliency = self.last_saliency_map[y_min_scaled:y_max_scaled,
                                                   x_min_scaled:x_max_scaled]

        # Compute normalized saliency
        mean_saliency = object_saliency.mean()
        max_saliency = self.last_saliency_map.max()

        if max_saliency > 0:
            return float(mean_saliency / max_saliency)
        return 0.0

    def observation(self, observation):
        """
        Create observation.

        Note: SARFA saliency maps should be generated explicitly via generate_saliency_map(),
        not automatically during gameplay (too slow).
        """
        # Don't auto-generate saliency during gameplay - too slow!
        # Users should call generate_saliency_map() explicitly when needed.
        return super().observation(observation)

