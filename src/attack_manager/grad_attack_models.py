# Copyright (c) Anish Acharya.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

import numpy as np
from typing import Dict
import warnings


class ByzAttack:
    """ This is the Base Class for Byzantine attack. """

    def __init__(self, attack_config: Dict, seed=1):
        self.attack_config = attack_config
        self.seed = seed
        self.attack_mode = self.attack_config.get('attack_mode', 'un_coordinated')
        self.attack_algorithm = self.attack_config.get('attack_model', None)
        self.frac_adv = self.attack_config.get('frac_adv', 0)

    def attack(self, g: np.array):
        pass

    def launch_attack(self, G: np.ndarray):
        np.random.seed(self.seed)
        max_adv = int(self.frac_adv * G.shape[0])
        if self.attack_mode == 'un_coordinated':
            for i in range(G.shape[0]):
                # Toss a coin
                if np.random.random_sample() < self.frac_adv:
                    perturbed_grad = self.attack(g=G[i, :])
                    G[i, :] = perturbed_grad
                    max_adv -= 1

                if max_adv == 0:
                    return G

                else:
                    continue
        elif self.attack_mode == 'coordinated':
            perturbed_grad = self.attack(g=G[0, :])
            for i in range(G.shape[0]):
                if np.random.random_sample() < self.frac_adv:
                    G[i, :] = perturbed_grad
                else:
                    continue

            # raise NotImplementedError
        return G


class DriftAttack(ByzAttack):
    """
    Implementation of the powerful drift attack algorithm proposed in:
    Ref: Gilad Baruch et.al. "A Little Is Enough: Circumventing Defenses For Distributed Learning" (NeurIPS 2019)
    https://github.com/moranant/attacking_distributed_learning
    """

    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)
        self.n_std = attack_config["attack_n_std"]

    def attack(self, byz_clients):
        if len(byz_clients) <= 1 or self.n_std == 0:
            warnings.warn(message='Drift Attack is implemented as a co-ordinated only attack,'
                                  'In the un-coordinated mode we are leaving the grads unchanged')
            return
        clients_grad = []
        for client in byz_clients:
            clients_grad.append(client.grad)
        grad_mean = np.mean(clients_grad, axis=0)
        grad_std = np.var(clients_grad, axis=0) ** 0.5
        # apply grad corruption = [ \mu - std * \sigma ]
        byz_grad = grad_mean[:] - self.n_std * grad_std[:]
        for client in byz_clients:
            client.grad = byz_grad


class Additive(ByzAttack):
    """
    Additive Gaussian Noise, scaled w.r.t the original values.
    Implementation of the attack mentioned (in un-coordinated setting only) in:
    Ref: Fu et.al. Attack-Resistant Federated Learning with Residual-based Re-weighting
    https://arxiv.org/abs/1912.11464.

    [Our Proposal] In Co-ordinated Mode: We take the mean of hem clients and generate noise based on the
    mean vector and make hem the clients grad = mean(grad_i) + noise.
    """

    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)

        self.rand_additive_attack_conf = attack_config.get("rand_additive_attack_conf", {})
        print(' Additive Noise Attack {} '.format(self.rand_additive_attack_conf))

        self.noise_dist = self.rand_additive_attack_conf["noise_dist"]

        # Gaussian Noise Model Configs
        self.attack_std = self.rand_additive_attack_conf.get("attack_std", 1)
        self.mean_shift = self.rand_additive_attack_conf.get("mean_shift", 1)

        # Uniform Noise Model Config
        self.noise_range = self.rand_additive_attack_conf.get("noise_range", [0, 1])

    def attack(self, g):
        if self.noise_dist == 'gaussian':
            noise = np.random.normal(loc=g*self.mean_shift,
                                     scale=self.attack_std,
                                     size=g.shape).astype(dtype=g.dtype)

        elif self.noise_dist == 'uniform':
            noise_ub = self.noise_range[1]
            noise_lb = self.noise_range[0]
            dist = noise_ub - noise_lb
            noise = np.random.random(g.shape) * dist + noise_lb
        else:
            raise NotImplementedError

        return g + noise


class Random(ByzAttack):
    """
    Random Gaussian Noise as used in the paper (in the uncoordinated setting)
    Ref: Cong et.al. Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance (ICML'19).

    [Our Proposal] In the co-ordinate setting hem the mal client's grad vectors are set to the same,
    drawn randomly from a Normal Distribution with zero mean and specified std
    """

    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)
        self.rand_additive_attack_conf = attack_config.get("rand_additive_attack_conf", {})
        print(' Additive Noise Attack {} '.format(self.rand_additive_attack_conf))

        self.noise_dist = self.rand_additive_attack_conf["noise_dist"]

        # Gaussian Noise Model Configs
        self.attack_std = self.rand_additive_attack_conf.get("attack_std", 1)
        self.mean_shift = self.rand_additive_attack_conf.get("mean_shift", 1)

        # Uniform Noise Model Config
        self.noise_range = self.rand_additive_attack_conf.get("noise_range", [0, 1])

    def attack(self, g):
        # apply gaussian noise (scaled appropriately)
        if self.noise_dist == 'gaussian':
            noise = np.random.normal(loc=g*self.mean_shift,
                                     scale=self.attack_std,
                                     size=g.shape).astype(dtype=g.dtype)

        elif self.noise_dist == 'uniform':
            dist = self.noise_range[1] - self.noise_range[0]
            min = self.noise_range[0]
            noise = np.random.random(g.shape) * dist + min
        else:
            raise NotImplementedError

        return noise


class BitFlipAttack(ByzAttack):
    """
    The bits that control the sign of the floating numbers are flipped, e.g.,
    due to some hardware failure. A faulty worker pushes the negative gradient instead
    of the true gradient to the servers.
    In Co-ordinated mode: one of the faulty gradients is copied to and overwrites the other faulty gradients,
    which means that hem the faulty gradients have the same value = - mean(g_i) ; i in byz clients

    Ref: Cong et.al. Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance (ICML'19).
    """

    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)
        self.sign_flip_conf = self.attack_config.get("sign_flip_conf", {})
        self.flip_scale = self.sign_flip_conf.get("flip_scale", 2)
        print(' Bit flip attack {} '.format(self.sign_flip_conf))

    def attack(self, g):
        return - self.flip_scale * g


class RandomSignFlipAttack(ByzAttack):
    """
    A faulty worker randomly (binomial) flips the sign of its each gradient co-ordinates
    In Co-ordinate Mode: We do the same to the mean of hem the byz workers and hem workers are
    assigned the same faulty gradient.
    Ref: Bernstein et.al. SIGNSGD WITH MAJORITY VOTE IS COMMUNICATION EFFICIENT AND FAULT TOLERANT ; (ICLR '19)
    """

    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)
        self.sign_flip_conf = self.attack_config.get("sign_flip_conf", {})
        self.flip_prob = self.sign_flip_conf.get("flip_prob", 0.5)

    def attack(self, g):
        faulty_grad = np.zeros_like(g)
        for i in range(0, len(g)):
            faulty_grad[i] = g[i] if np.random.random() > self.flip_prob else -g[i]

        return faulty_grad


