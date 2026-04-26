"""Shared SARFA math helpers used by multiple wrappers."""

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy


def cross_entropy(original_output, perturbed_output, action_index):
    """Compute SARFA K-term from action distributions without chosen action."""
    p = original_output[:action_index]
    p = np.append(p, original_output[action_index + 1 :])
    p = softmax(p)

    new_p = perturbed_output[:action_index]
    new_p = np.append(new_p, perturbed_output[action_index + 1 :])
    new_p = softmax(new_p)

    kl_div = entropy(p, new_p)
    return 1.0 / (1.0 + kl_div)


def sarfa_saliency(original_output, perturbed_output, action_index):
    """Calculate SARFA saliency score for one perturbation."""
    original_output = np.squeeze(original_output)
    perturbed_output = np.squeeze(perturbed_output)
    d_p = softmax(original_output)[action_index] - softmax(perturbed_output)[action_index]
    if d_p <= 0:
        return 0.0
    k = cross_entropy(original_output, perturbed_output, action_index)
    return (2 * k * d_p) / (k + d_p)
