"""Clean LayerNorm implementation (numpy).

This is a minimal, easy-to-read LayerNorm used for learning and reference.
"""

import numpy as np


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.dim = dim
        self.eps = eps
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)

    def __call__(self, x):
        """Apply layer normalization over the last axis of `x`.

        Args:
            x: np.ndarray with shape (..., dim)

        Returns:
            normalized array with same shape as `x`.
        """
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
