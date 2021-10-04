# Copyright (c) Anish Acharya.
# Licensed under the MIT License

import numpy as np
from .base_gar import GAR
from typing import List

"""
Ghosh et.al. Communication-Efficient and Byzantine-Robust Distributed Learning with Error Feedback
"""


class NormClipping(GAR):

    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)
        # Find Number of top norms to drop
        self.alpha = self.aggregation_config.get("norm_clip_config", {}).get("alpha", 0.1)
        self.k = None

    def aggregate(self, G: np.ndarray, ix: List[int] = None, axis=0) -> np.ndarray:
        # Compute norms of each gradient vector
        # norm_dist = np.linalg.norm(G, axis=1)

        if self.k is None:
            self.k = int(G.shape[0] * self.alpha)
            print('Norm clipping {} clients'.format(self.k))

        norms = np.sqrt(np.einsum('ij,ij->i', G, G))
        top_k_indices = np.argsort(np.abs(norms))[::-1][:self.k]

        # set weights of them to 0 filtering k top ones based on norm
        alphas = np.ones(G.shape[0]) * (1 / (G.shape[0] - self.k))
        alphas[top_k_indices] = 0
        agg_grad = self.weighted_average(stacked_grad=G, alphas=alphas)

        if ix is not None:
            return agg_grad[ix]
        else:
            return agg_grad
