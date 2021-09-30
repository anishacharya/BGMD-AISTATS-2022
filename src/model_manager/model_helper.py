from .cnn import LeNet
from typing import Dict


def get_model(learner_config: Dict, data_config: Dict, seed=1):
    """ wrapper to return appropriate model class """
    net = learner_config.get("net", 'lenet')
    print('Loading Model: {}'.format(net))
    print('----------------------------')
    nc = data_config.get("num_channels", 1)
    shape = data_config.get("shape", [28, 28])

    if net == 'lenet':
        model = LeNet(nc=nc, nh=shape[0], hw=shape[1], num_classes=data_config["num_labels"], seed=seed)
    else:
        raise NotImplementedError

    print(model)
    return model

