from .vision_datasets import *
from .nlp_datasets import *


def process_data(data_config: Dict):
    data_set = data_config["data_set"]
    # if data_set == 'cifar10':
    #     return CIFAR10(data_config=data_config)
    if data_set == 'mnist':
        return MNIST(data_config=data_config)
    elif data_set == 'fashion_mnist':
        return FashionMNIST(data_config=data_config)
    # elif data_set == 'imagenet':
    #     return ImageNet(data_config=data_config)
    # elif data_set == 'extended_mnist':
    #     return ExtendedMNIST(data_config=data_config)
    elif data_set == 'sst':
        return IMDB(data_config=data_config)
    else:
        raise NotImplemented

