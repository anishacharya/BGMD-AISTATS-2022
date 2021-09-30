import torch.utils.data

from torchvision import datasets, transforms
from typing import Dict
import os

curr_dir = os.path.dirname(__file__)
root = os.path.join(curr_dir, './data/')


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


class VisionDataManager:
    """
    Base Class for Vision Data Readers
    """

    def __init__(self, data_config: Dict, transformation=None):
        self.data_config = data_config
        self.transform = transformation

    def download_data(self):
        """ Downloads Data and Apply appropriate Transformations . returns train, test dataset """
        raise NotImplementedError("This method needs to be implemented")

    @staticmethod
    def _get_common_data_trans(_train_dataset):
        """ Implements a simple way to compute train and test transform that usually works """
        try:
            mean = [_train_dataset.data.float().mean(axis=(0, 1, 2)) / 255]
            std = [_train_dataset.data.float().std(axis=(0, 1, 2)) / 255]
        except:
            mean = _train_dataset.data.mean(axis=(0, 1, 2)) / 255
            std = _train_dataset.data.std(axis=(0, 1, 2)) / 255

        return mean, std


class MNIST(VisionDataManager):
    def __init__(self, data_config: Dict):
        VisionDataManager.__init__(self, data_config=data_config)

    def download_data(self, seed=1):
        torch.manual_seed(seed)
        _train_dataset = datasets.MNIST(root=root, download=True)

        # Whiten the Data
        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        _train_dataset = datasets.MNIST(root=root, download=True, transform=train_trans)
        _test_dataset = datasets.MNIST(root=root, download=True, train=False, transform=test_trans)

        return _train_dataset, _test_dataset


class FashionMNIST(VisionDataManager):
    def __init__(self, data_config: Dict):
        VisionDataManager.__init__(self, data_config=data_config)

    def download_data(self, seed=1):
        torch.manual_seed(seed)
        _train_dataset = datasets.FashionMNIST(root=root, download=True)

        # Whiten the Data
        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        _train_dataset = datasets.FashionMNIST(root=root, download=True, transform=train_trans)
        _test_dataset = datasets.FashionMNIST(root=root, download=True, train=False, transform=test_trans)

        return _train_dataset, _test_dataset


class ExtendedMNIST(VisionDataManager):
    def __init__(self, data_config: Dict):
        VisionDataManager.__init__(self, data_config=data_config)

    def download_data(self, seed=1):
        torch.manual_seed(seed)
        _train_dataset = datasets.EMNIST(root=root, download=True, split='balanced')

        # Whiten the Data
        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        _train_dataset = datasets.EMNIST(root=root, download=True, transform=train_trans, split='balanced')
        _test_dataset = datasets.EMNIST(root=root, download=True, train=False, transform=test_trans, split='balanced')

        return _train_dataset, _test_dataset


class CIFAR10(VisionDataManager):
    def __init__(self, data_config: Dict):
        VisionDataManager.__init__(self, data_config=data_config)

    def download_data(self):
        _train_dataset = datasets.CIFAR10(root=root, download=True)

        # Whiten the Data
        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        _train_dataset = datasets.CIFAR10(root=root, download=True, transform=train_trans)
        _test_dataset = datasets.CIFAR10(root=root, download=True, train=False, transform=test_trans)

        return _train_dataset, _test_dataset


class ImageNet(VisionDataManager):
    def __init__(self, data_config: Dict):
        VisionDataManager.__init__(self, data_config=data_config)

    # noinspection PyTypeChecker
    def download_data(self):
        _train_dataset = datasets.ImageNet(root=root, download=True)

        # Whiten the Data
        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        _train_dataset = datasets.ImageNet(root=root, download=True, transform=train_trans)
        _test_dataset = datasets.ImageNet(root=root, download=True, train=False, transform=test_trans)

        return _train_dataset, _test_dataset
