from .vision_datasets import *


def get_data_manager(data_config: Dict):
    data_set = data_config["data_set"]
    if data_set == 'cifar10':
        return CIFAR10(data_config=data_config)
    elif data_set == 'mnist':
        return MNIST(data_config=data_config)
    elif data_set == 'fashion_mnist':
        return FashionMNIST(data_config=data_config)
    elif data_set == 'imagenet':
        return ImageNet(data_config=data_config)
    elif data_set == 'extended_mnist':
        return ExtendedMNIST(data_config=data_config)
    else:
        raise NotImplemented

