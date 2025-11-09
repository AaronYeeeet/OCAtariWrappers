import numpy as np
import torch
from ocatari_wrappers.masked_dqn import MaskedBaseWrapper


class GradientSaliencyWrapper(MaskedBaseWrapper):
    def __init__(self, env, trained_model=None, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.model = trained_model
        self.gradient_map = None

        if trained_model is not None:
            self.device = next(trained_model.parameters()).device
            print("✅ GradientSaliencyWrapper: Using gradient-based saliency")
        else:
            self.device = torch.device('cpu')
            print("⚠️  GradientSaliencyWrapper: No model provided, using uniform saliency")

    def observation(self, observation):
        #only compute saliency map when buffer is full
        if self.model is not None and len(self._buffer) == self.buffer_window_size:
            self._compute_saliency_map()
        return super().observation(observation)

    def _compute_saliency_map(self):
        current_obs = np.asarray(self._buffer)  #(4, 84, 84)

        # make the model store the gradients
        observations_tensor = torch.FloatTensor(current_obs).unsqueeze(0).to(self.device)
        observations_tensor.requires_grad = True

        # forward pass
        output = self.model.get_value(observations_tensor)

        # backward pass + gradient saving
        output.sum().backward() # writes gradients into observations_tensor.grad
        gradients = observations_tensor.grad.abs().cpu().numpy()[0]

        self.gradient_map = gradients.mean(axis=0)  # mean over last observations


    def set_value(self, y_min, y_max, x_min, x_max, o):
        saliency = self._get_object_saliency(y_min, y_max, x_min, x_max) #arg is box of object
        intensity = int(255 * np.clip(saliency, 0.0, 1.0))
        self.state[0, y_min:y_max, x_min:x_max].fill(intensity)         # fill the object box with calculated intensity

    def _get_object_saliency(self, y_min, y_max, x_min, x_max):
        # works only in frames with full buffer, so skip otherwise
        if self.gradient_map is None:
            return 0

        # adjust 210, 160 to 84, 84
        height_orig, width_orig = self.state.shape[1], self.state.shape[2]
        height_grad, width_grad = self.gradient_map.shape
        y_min_scaled = int(y_min * height_grad / height_orig)
        y_max_scaled = int(y_max * height_grad / height_orig)
        x_min_scaled = int(x_min * width_grad / width_orig)
        x_max_scaled = int(x_max * width_grad / width_orig)

        # get the gradients
        object_gradients = self.gradient_map[y_min_scaled:y_max_scaled,
                                              x_min_scaled:x_max_scaled]

        # compute the saliency
        mean_gradient = object_gradients.mean()
        max_gradient = self.gradient_map.max()
        saliency = mean_gradient / max_gradient

        return float(saliency)





