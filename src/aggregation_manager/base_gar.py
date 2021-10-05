# Copyright (c) Anish Acharya.
# Licensed under the MIT License

import numpy as np
import torch
from typing import List, Dict
from src.compression_manager import SparseApproxMatrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAR:
    """
    This is the base class for hem the implemented GAR
    """

    def __init__(self, aggregation_config):
        self.aggregation_config = aggregation_config
        self.current_losses = []
        self.agg_time = 0
        self.num_iter = 0  # usually if Sub routine has iters ex - GM

    def aggregate(self, G: np.ndarray, ix: List[int] = None, axis=0) -> np.ndarray:
        """
        G: Gradient Matrix where each row is a gradient vector (g_i)
        ix: Columns specified to be aggregated on (if None done on full dimension)
        """
        raise NotImplementedError

    def block_descent_aggregate(self, sparse_approx_config: Dict, G: np.ndarray):
        sparse_rule = sparse_approx_config.get('rule', None)
        sparse_selection = SparseApproxMatrix(conf=sparse_approx_config) \
            if sparse_rule in ['active_norm', 'random'] else None
        I_k = None
        if sparse_selection is not None:
            G, I_k = sparse_selection.compress(G=G, lr=1)
        agg_g = self.aggregate(G=G, ix=I_k)
        return agg_g

    @staticmethod
    def weighted_average(stacked_grad: np.ndarray, alphas=None):
        """
        Implements weighted average of grad vectors stacked along rows of G
        If no weights are supplied then its equivalent to simple average
        """
        if alphas is None:
            agg_grad = np.mean(stacked_grad, axis=0, dtype=stacked_grad.dtype)
        else:
            assert len(alphas) == stacked_grad.shape[0]
            agg_grad = np.matmul(alphas, stacked_grad, dtype=stacked_grad.dtype)
        return agg_grad
