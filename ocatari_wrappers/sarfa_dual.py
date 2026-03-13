"""Shared SARFA dual wrapper implementations (5-channel and 8-channel)."""

from collections import deque

import numpy as np
import torch
from gymnasium import spaces

from ocatari_wrappers.masked_dqn import MaskedBaseWrapper
from ocatari_wrappers.sarfa_common import cross_entropy, sarfa_saliency


class _SarfaDualBase(MaskedBaseWrapper):
	"""Base class with shared SARFA computation and rendering helpers."""

	def __init__(
		self,
		env,
		trained_model=None,
		sarfa_compute_interval=4,
		compute_every_step=None,
		*args,
		**kwargs,
	):
		super().__init__(env, *args, **kwargs)
		self.model = trained_model

		if compute_every_step is not None:
			self.sarfa_compute_interval = 1 if compute_every_step else 4
		else:
			self.sarfa_compute_interval = max(1, int(sarfa_compute_interval))

		self.steps_since_sarfa = 0
		self.sarfa_map = None
		self.min_visible = 0
		self._cached_sarfa_frame = np.zeros((84, 84), dtype=np.uint8)

	def set_model(self, model):
		"""Allows injecting the agent after environment creation."""
		self.model = model

	def set_value(self, y_min, y_max, x_min, x_max, o):
		"""Set binary mask value (white = 255) for objects in the state buffer."""
		self.state[0, y_min:y_max, x_min:x_max].fill(255)

	def _iter_valid_boxes(self):
		for obj in self.env.objects:  # noqa: OCAtari in the env stack
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

			yield y_min, y_max, x_min, x_max

	def _forward_logits(self, obs_batch_uint8):
		device = next(self.model.parameters()).device
		with torch.no_grad():
			obs_tensor = torch.as_tensor(obs_batch_uint8, dtype=torch.float32, device=device) / 255.0
			hidden = self.model.network(obs_tensor)
			logits = self.model.actor(hidden)
			return logits.detach().cpu().numpy()

	def _compute_sarfa_map(self, current_obs):
		original_output = self._forward_logits(np.expand_dims(current_obs, axis=0))
		action_index = int(np.argmax(original_output))

		sarfa_map = np.zeros((84, 84), dtype=np.float32)
		object_mask = np.zeros((84, 84), dtype=bool)
		valid_boxes = list(self._iter_valid_boxes())

		if not valid_boxes:
			return sarfa_map, object_mask

		perturbed_batch = np.repeat(np.expand_dims(current_obs, axis=0), len(valid_boxes), axis=0)
		for i, (y_min, y_max, x_min, x_max) in enumerate(valid_boxes):
			perturbed_batch[i, :, y_min:y_max, x_min:x_max] = 0

		perturbed_outputs = self._forward_logits(perturbed_batch)

		for (y_min, y_max, x_min, x_max), perturbed_output in zip(valid_boxes, perturbed_outputs):
			score = sarfa_saliency(original_output, perturbed_output, action_index)
			sarfa_map[y_min:y_max, x_min:x_max] = score
			object_mask[y_min:y_max, x_min:x_max] = True

		max_score = sarfa_map.max()
		if max_score > 0:
			sarfa_map /= max_score

		return sarfa_map, object_mask

	def _update_cached_sarfa_frame(self, current_obs):
		sarfa_map, object_mask = self._compute_sarfa_map(current_obs)
		self.sarfa_map = sarfa_map

		frame = (sarfa_map * 255).astype(np.uint8)
		frame[object_mask] = np.maximum(frame[object_mask], self.min_visible)
		self._cached_sarfa_frame[:] = frame

	def _clear_sarfa_state(self):
		self.steps_since_sarfa = 0
		self.sarfa_map = None
		self._cached_sarfa_frame.fill(0)

	def _ready_for_sarfa(self, binary_obs):
		return self.model is not None and binary_obs.shape[0] == self._buffer.maxlen


class SarfaDualWrapperFive(_SarfaDualBase):
	"""5-channel wrapper: binary stack + current SARFA frame (typically 4 + 1)."""

	def __init__(self, env, trained_model=None, *args, **kwargs):
		super().__init__(env, trained_model=trained_model, *args, **kwargs)
		self._binary_channels = self._buffer.maxlen
		self._combined_obs = np.zeros((self._binary_channels + 1, 84, 84), dtype=np.uint8)
		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=(self._binary_channels + 1, 84, 84),
			dtype=np.uint8,
		)

	def observation(self, observation):
		binary_obs = super().observation(observation)
		n_binary = binary_obs.shape[0]

		self._combined_obs[:n_binary] = binary_obs
		self._combined_obs[n_binary] = self._cached_sarfa_frame

		if self._ready_for_sarfa(binary_obs):
			self.steps_since_sarfa += 1
			if self.steps_since_sarfa >= self.sarfa_compute_interval:
				self._update_cached_sarfa_frame(self._combined_obs)
				self.steps_since_sarfa = 0

		return self._combined_obs

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		self._clear_sarfa_state()

		final_obs = self.observation(obs)
		for _ in range(max(0, self.buffer_window_size - 1)):
			final_obs = self.observation(obs)

		return final_obs, info


class SarfaDualWrapperEight(_SarfaDualBase):
	"""8-channel wrapper: binary stack + SARFA frame stack (typically 4 + 4)."""

	def __init__(self, env, trained_model=None, *args, **kwargs):
		super().__init__(env, trained_model=trained_model, *args, **kwargs)
		self.sarfa_frame_buffer = deque(maxlen=self.buffer_window_size)
		self._binary_channels = self._buffer.maxlen
		self._combined_obs = np.zeros((self._binary_channels + self.buffer_window_size, 84, 84), dtype=np.uint8)
		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=(self._binary_channels + self.buffer_window_size, 84, 84),
			dtype=np.uint8,
		)

	def observation(self, observation):
		binary_obs = super().observation(observation)
		n_binary = binary_obs.shape[0]

		self.sarfa_frame_buffer.append(self._cached_sarfa_frame.copy())

		self._combined_obs[:n_binary] = binary_obs
		for i, frame in enumerate(self.sarfa_frame_buffer):
			self._combined_obs[n_binary + i] = frame

		if self._ready_for_sarfa(binary_obs) and len(self.sarfa_frame_buffer) == self.buffer_window_size:
			self.steps_since_sarfa += 1
			if self.steps_since_sarfa >= self.sarfa_compute_interval:
				self._update_cached_sarfa_frame(self._combined_obs)
				self.steps_since_sarfa = 0

		return self._combined_obs

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		self._clear_sarfa_state()
		self.sarfa_frame_buffer.clear()
		for _ in range(self.buffer_window_size):
			self.sarfa_frame_buffer.append(np.zeros((84, 84), dtype=np.uint8))

		final_obs = self.observation(obs)
		for _ in range(max(0, self.buffer_window_size - 1)):
			final_obs = self.observation(obs)

		return final_obs, info

