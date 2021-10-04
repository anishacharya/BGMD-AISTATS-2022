# Copyright (c) Anish Acharya
# Licensed under the MIT License

import numpy as np
import math
from .compression_base import GradientCompression


class Adaptive(GradientCompression):
    def __init__(self, conf):
        GradientCompression.__init__(self, conf=conf)

    def compress(self, g: np.ndarray, lr=1):
        pass


class Full(GradientCompression):
    def __init__(self, conf):
        GradientCompression.__init__(self, conf=conf)

    def compress(self, g: np.ndarray, lr=1):
        return g


class Top(GradientCompression):
    """
    Implements Top K sparse compression
    Also supports the following memory mechanisms:
    'ef' : error feedback (RE: Stich et.al. Sparsified SGD with Memory; NeurIPS'18)
    """
    def __init__(self, conf):
        GradientCompression.__init__(self, conf=conf)
        self.beta = conf.get('sampling_fraction', 1)

    def compress(self, g: np.ndarray, lr=1) -> np.ndarray:
        if self.beta == 1:
            return g
        # Add memory feedback if enabled
        g = self.memory_feedback(g=g, lr=lr)
        # Compression
        self.compressed_g = np.zeros_like(g)
        num_coordinates_to_keep = math.floor(self.beta * len(g))
        indices = np.argsort(np.abs(g))[::-1][:num_coordinates_to_keep]
        self.compressed_g[indices] = g[indices]
        # update memory
        self.memory_update(g=g, lr=lr)
        return self.compressed_g


class Rand(GradientCompression):
    def __init__(self, conf):
        GradientCompression.__init__(self, conf=conf)
        self.k = conf.get('frac_coordinates_to_keep', 0.1)

    def compress(self, g: np.ndarray, lr=1) -> np.ndarray:
        if self.residual_error is None:
            self.residual_error = np.zeros_like(g)

        g = (lr * g) + self.residual_error

        compressed_g = np.zeros_like(g)
        num_coordinates_to_keep = round(self.k * len(g))
        indices = np.random.choice(a=np.arange(len(g)),
                                   size=num_coordinates_to_keep)
        compressed_g[indices] = g[indices]

        if self.ef is not None:
            self.residual_error = g - compressed_g
            compressed_g /= lr

        return compressed_g


class Q(GradientCompression):
    def __init__(self, conf):
        GradientCompression.__init__(self, conf=conf)
        self.q = conf.get('bits', 2)

    def compress(self, g: np.ndarray, lr=1) -> np.ndarray:
        s = 2 ** self.q
        # levels = np.arange(s+1)/s

        # compressed_g = np.zeros_like(g)
        g_norm = np.linalg.norm(g)

        if g_norm == 0:
            return np.zeros_like(g).astype(np.float16)  # compressed_g

        # g_sign = np.sign(g)
        # g_val = np.abs(g)
        g_levels = np.floor((np.abs(g) / g_norm) * s)
        # g_probs = ((g_val/g_norm)*s - g_levels)
        g_probs = (np.abs(g) / g_norm) * s - g_levels  # np.floor((np.abs(g)/g_norm)*s)

        zeta = np.random.binomial(1, 1.0 - g_probs, len(g))
        val = (zeta * (g_levels / s) + (1.0 - zeta) * ((g_levels + 1) / s)).astype(np.float16)
        # compressed_g = g_norm*np.sign(g)*val
        return (g_norm * np.sign(g) * val).astype(np.float16)