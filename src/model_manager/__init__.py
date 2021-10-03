from .model_helper import *
from torch import optim
from .loss_functions import *
from .vision_cnn import *

import torch.nn as nn


def get_loss(loss: str):
    """ returns: per sample loss """
    if loss == 'mse':
        return nn.MSELoss(reduction='none')
    elif loss == 'ce':
        return nn.CrossEntropyLoss(reduction='none')
    elif loss == 'bce':
        return nn.BCELoss(reduction='none')
    else:
        raise NotImplementedError


def get_optimizer(params, optimizer_config: Dict = None):
    if optimizer_config is None:
        optimizer_config = {}
    opt_alg = optimizer_config.get('optimizer', 'SGD')

    if opt_alg == 'SGD':
        return optim.SGD(params=params,
                         lr=optimizer_config.get('lr0', 1),
                         momentum=optimizer_config.get('momentum', 0),
                         weight_decay=optimizer_config.get('reg', 0),
                         nesterov=optimizer_config.get('nesterov', False),
                         dampening=optimizer_config.get('damp', 0))
    elif opt_alg == 'Adam':
        return optim.Adam(params=params,
                          lr=optimizer_config.get('lr0', 1),
                          betas=optimizer_config.get('betas', (0.9, 0.999)),
                          eps=optimizer_config.get('eps', 1e-08),
                          weight_decay=optimizer_config.get('reg', 0.05),
                          amsgrad=optimizer_config.get('amsgrad', False))

    else:
        raise NotImplementedError
