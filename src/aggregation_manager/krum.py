# Copyright (c) Anish Acharya.
# Licensed under the MIT License

import numpy as np
from .base_gar import GAR


class Krum(GAR):
    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)

    def aggregate(self, G: np.ndarray) -> np.ndarray:
        dist = self.get_krum_dist(G=G)
        krum_conf = self.aggregation_config.get("krum_config", {})

        m = int(krum_conf.get("krum_frac", 0.3) * G.shape[0])
        min_score = 1e10
        optimal_client_ix = -1

        for ix in range(G.shape[0]):
            curr_dist = dist[ix, :]
            curr_dist = np.sort(curr_dist)
            curr_score = sum(curr_dist[:m])
            if curr_score < min_score:
                min_score = curr_score
                optimal_client_ix = ix
        krum_grad = G[optimal_client_ix, :]
        return krum_grad

    @staticmethod
    def get_krum_dist(G: np.ndarray) -> np.ndarray:
        """ Computes distance between each pair of client based on grad value """
        dist = np.zeros((G.shape[0], G.shape[0]))
        for i in range(G.shape[0]):
            for j in range(i):
                dist[i][j] = dist[j][i] = np.linalg.norm(G[i, :] - G[j, :])
        return dist
