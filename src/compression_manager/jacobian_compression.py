# Copyright (c) Anish Acharya.
# Licensed under the MIT License

"""
Here we explore different co-ordinate sampling ideas.
Essentially, the idea is to instead of taking hem the coordinates sub-sample
pre-defined k subset of co-ordinates and aggregate only along these directions.

--- This can be thought of a column sampling of matrix G where each row
corresponds to g_i i.e. gradient vector computed on batch / client i
"""

import numpy as np
np.random.seed(1)


class JacobianCompression:
    """
    This is Base Class for hem Jacobian Compression.
    --- This can be thought of compression of matrix G where each row
    corresponds to g_i i.e. gradient vector computed on batch / client i
    """
    def __init__(self, conf):
        self.conf = conf
        self.compression_rule = self.conf.get('rule', None)
        self.memory_algo = self.conf.get('memory_algo', None)
        axis = self.conf.get('axis', 'dim')  # 0: column / dimension , 1: row / samples(clients)
        if axis == 'dim':
            self.axis = 0
        elif axis == 'n':
            self.axis = 1
        else:
            raise ValueError

        self.n = None
        self.d = None
        self.G_sparse = None

        self.residual_error = None
        self.normalized_residual = 0

    def compress(self, G: np.ndarray, lr=1) -> [np.ndarray, np.ndarray]:
        raise NotImplementedError("This method needs to be implemented for each Compression Algorithm")

    def memory_feedback(self, G: np.ndarray, lr=1) -> np.ndarray:
        """ Chosen Form of memory is added to Jacobian as feedback """
        if not self.memory_algo:
            return G
        elif self.memory_algo == 'ef':
            if self.residual_error is None:
                self.residual_error = np.zeros((self.n, self.d), dtype=G[0, :].dtype)
            return (lr * G) + self.residual_error
        else:
            raise NotImplementedError

    def memory_update(self, G: np.ndarray, lr=1):
        """ update the memory vector """
        if not self.memory_algo:
            return
        elif self.memory_algo == 'ef':
            delta = G - self.G_sparse
            memory = np.mean(delta, axis=0)
            self.residual_error = np.tile(memory, (G.shape[0], 1))
            self.G_sparse /= lr
        else:
            raise NotImplementedError


class SparseApproxMatrix(JacobianCompression):
    def __init__(self, conf):
        JacobianCompression.__init__(self, conf=conf)
        self.frac = conf.get('sampling_fraction', 1)  # fraction of ix to sample
        self.k = None  # Number of ix ~ to be auto populated

    def compress(self, G: np.ndarray, lr=1) -> np.ndarray:

        self.G_sparse = np.zeros_like(G)
        if self.compression_rule not in ['active_norm_sampling', 'random_sampling']:
            raise NotImplementedError

        G = self.memory_feedback(G=G, lr=lr)

        # for the first run compute k and residual error
        if self.k is None:
            self.n, self.d = G.shape
            if self.frac < 0:
                raise ValueError
            elif self.axis == 0:
                self.k = int(self.frac * self.d) if self.frac > 0 else 1
                print('Sampling {} coordinates out of {}'.format(self.k, self.d))
            elif self.axis == 1:
                self.k = int(self.frac * self.n) if self.frac > 0 else 1
                print('Sampling {} samples out of {}'.format(self.k, self.n))

        # Invoke Sampling algorithm
        if self.compression_rule == 'active_norm_sampling':
            I_k = self._active_norm_sampling(G=G)
        elif self.compression_rule == 'random_sampling':
            I_k = self._random_sampling(d=self.d if self.axis == 0 else self.n)
        else:
            raise NotImplementedError

        if self.axis == 0:
            self.G_sparse[:, I_k] = G[:, I_k]
        elif self.axis == 1:
            self.G_sparse[I_k, :] = G[I_k, :]
        else:
            raise ValueError

        self.memory_update(G=G, lr=lr)

        return I_k

    # Implementation of different "Matrix Sparse Approximation" strategies
    def _random_sampling(self, d) -> np.ndarray:
        """
        Implements Random (Gauss Siedel) subset Selection
        """
        all_ix = np.arange(d)
        I_k = np.random.choice(a=all_ix,
                               size=self.k,
                               replace=False)

        return I_k

    def _active_norm_sampling(self, G: np.ndarray) -> np.ndarray:
        """
        Implements Gaussian Southwell Subset Selection / Active norm sampling
        Ref: Drineas, P., Kannan, R., and Mahoney, M. W.  Fast monte carlo algorithms for matrices:
        Approximating matrix multiplication. SIAM Journal on Computing, 36(1):132–157, 2006
        """
        # Exact Implementation ~ O(d log d)
        # norm_dist = G.sum(axis=self.axis)
        # norm_dist = np.square(norm_dist)
        norm_dist = np.linalg.norm(G, axis=self.axis)
        norm_dist /= norm_dist.sum()
        sorted_ix = np.argsort(norm_dist)[::-1]

        I_k = sorted_ix[:self.k]

        mass_explained = np.sum(norm_dist[I_k])
        self.normalized_residual = mass_explained

        return I_k
