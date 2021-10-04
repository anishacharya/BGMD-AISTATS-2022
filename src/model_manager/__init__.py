from torch import optim
from .loss_functions import *
from .vision_cnn import *
from .fasttext import *

import torch.nn as nn
from typing import Dict


def get_loss(loss_fn: str):
    """ returns: per sample loss """
    if loss_fn == 'mse':
        return nn.MSELoss(reduction='none')
    elif loss_fn == 'ce':
        return nn.CrossEntropyLoss(reduction='none')
    elif loss_fn == 'bce':
        return nn.BCELoss(reduction='none')
    else:
        raise NotImplementedError


def get_model(learner_config: Dict, data_config: Dict, seed=1, additional_conf=None):
    """ wrapper to return appropriate model class """
    if additional_conf is None:
        additional_conf = {}
    net = learner_config.get("net", 'lenet')
    print('Loading Model: {}'.format(net))
    print('----------------------------')

    if net == 'lenet':
        nc = data_config.get("num_channels", 1)
        shape = data_config.get("shape", [28, 28])
        model = LeNet(nc=nc, nh=shape[0], hw=shape[1], num_classes=data_config["num_labels"], seed=seed)

    elif net == 'fasttext':
        vs = additional_conf['vocab_size']
        ed = additional_conf['embedding_dim']
        od = additional_conf['output_dim']
        px = additional_conf['pad_idx']
        model = FastText(vocab_size=vs, embedding_dim=ed, output_dim=od, pad_idx=px)
    else:
        raise NotImplementedError

    print(model)
    return model


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


def get_scheduler(optimizer, lrs_config: Dict = None):
    lrs = lrs_config.get('lrs')

    if lrs == 'step':
        return optim.lr_scheduler.StepLR(optimizer=optimizer,
                                         step_size=lrs_config.get('step_size', 10),
                                         gamma=lrs_config.get('gamma', 0.9))
    elif lrs == 'multi_step':
        return optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                              milestones=lrs_config.get('milestones', [100]),
                                              gamma=lrs_config.get('gamma', 0.5),
                                              last_epoch=lrs_config.get('last_epoch', -1))
    elif lrs == 'exp':
        return optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                gamma=lrs_config.get('gamma', 0.5))
    elif lrs == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100)
    elif lrs == 'cyclic':
        max_lr = lrs_config.get('lr0', 0.001)
        base_lr = 0.1 * max_lr
        return optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                           base_lr=base_lr,
                                           max_lr=max_lr,
                                           step_size_up=lrs_config.get('step_size_up', 100))
    else:
        return None


def take_lrs_step(clients):
    for client in clients:
        if client.lrs:
            client.lrs.step()

    current_lr = clients[0].optimizer.param_groups[0]['lr']

    if len(clients) > 1:
        assert (clients[0].optimizer.param_groups[0]['lr'] == clients[1].optimizer.param_groups[0]['lr'])

    return current_lr
