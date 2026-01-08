"""Simple token + positional embeddings (numpy).

Readable reference implementation for learning purposes.
"""

import numpy as np


class Embeddings:
    def __init__(self, vocab_size: int, dim: int):
        self.vocab_size = vocab_size
        self.dim = dim
        self.token_emb = np.random.randn(vocab_size, dim).astype(np.float32) * 0.01

    def token(self, token_ids):
        """Return token embeddings for token id array of shape (batch, seq)."""
        return self.token_emb[token_ids]

    def add_positional(self, x):
        """Add sinusoidal positional encodings to `x`.

        Args:
            x: np.ndarray shape (batch, seq, dim)
        """
        batch, seq_len, dim = x.shape
        assert dim == self.dim
        pos = np.arange(seq_len)[:, None]
        i = np.arange(dim)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dim))
        angle_rads = pos * angle_rates
        pos_encoding = np.zeros((seq_len, dim), dtype=np.float32)
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return x + pos_encoding[None, :, :]
