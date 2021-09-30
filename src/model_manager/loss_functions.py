import torch
import math
import numpy as np


def loss_sampler(loss, sampler=None, evaluate=False, beta=0.1):
    """
    Implementation of Different Loss Modifications ~
    See Loss Pipeline for usage example
    """
    batch_loss = torch.mean(loss)  # loss over all samples

    if sampler is None or evaluate:
        return batch_loss, None
    else:
        # Do Loss Sampling only during Training
        k = math.ceil((1 - beta) * len(loss))
        if sampler == 'top_loss':
            # Implements : Ordered SGD: A New Stochastic Optimization Framework for Empirical Risk Minimization
            # Kawaguchi, Kenji and Lu, Haihao; AISTATS 2020
            top_k_loss, top_k_ix = torch.topk(loss, k, sorted=False)
            return batch_loss, torch.mean(top_k_loss)

        elif sampler == 'rand_loss':
            # Random Sampling
            rand_k_ix = np.random.choice(len(loss), k)
            rand_k_loss = loss[rand_k_ix]
            return batch_loss, torch.mean(rand_k_loss)

        else:
            raise NotImplementedError
