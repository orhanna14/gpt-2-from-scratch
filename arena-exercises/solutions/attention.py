"""Scaled dot-product attention (numpy).

Simple, easy-to-read version for learning and experimentation.
"""

import numpy as np


def scaled_dot_product_attention(q, k, v, mask=None):
    """Compute attention outputs.

    Args:
        q, k, v: arrays with shape (batch, heads, seq, depth)
        mask: optional boolean mask with shape broadcastable to scores

    Returns:
        tuple(attended_values, attention_weights)
    """
    dk = q.shape[-1]
    scores = np.matmul(q, k.swapaxes(-2, -1)) / np.sqrt(dk)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    # stable softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(scores - scores_max)
    weights = exp / (np.sum(exp, axis=-1, keepdims=True) + 1e-9)
    out = np.matmul(weights, v)
    return out, weights
