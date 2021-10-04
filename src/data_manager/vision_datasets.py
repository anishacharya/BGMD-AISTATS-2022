import torch.utils.data

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Dict
import os

curr_dir = os.path.dirname(__file__)
root = os.path.join(curr_dir, './data/')


class VisionDataManager:
    """
    Base Class for Vision Data Readers
    """

    def __init__(self, data_config):
        self.data_config = data_config
        self.additional_model_conf = {}

    def get_data_iterator(self):
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
    def __init__(self, data_config):
        VisionDataManager.__init__(self, data_config=data_config)

    def get_data_iterator(self, seed=1):
        torch.manual_seed(seed)
        _train_dataset = datasets.MNIST(root=root, download=True)

        # Whiten the Data
        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        # Download Dataset
        _train_dataset = datasets.MNIST(root=root, download=True, transform=train_trans)
        _test_dataset = datasets.MNIST(root=root, download=True, train=False, transform=test_trans)

        # create iterators
        tr_batch_size = self.data_config.get('train_batch_size', 1)
        test_batch_size = self.data_config.get('test_batch_size', 512)
        train_loader = DataLoader(dataset=_train_dataset, batch_size=tr_batch_size, shuffle=True)
        print('Num of Batches in Train Loader = {}'.format(len(train_loader)))
        test_loader = DataLoader(dataset=_test_dataset, batch_size=test_batch_size)

        return train_loader, test_loader


class FashionMNIST(VisionDataManager):
    def __init__(self, data_config):
        VisionDataManager.__init__(self, data_config=data_config)

    def get_data_iterator(self, seed=1):
        torch.manual_seed(seed)
        _train_dataset = datasets.FashionMNIST(root=root, download=True)

        # Whiten the Data
        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        # Download Dataset
        _train_dataset = datasets.FashionMNIST(root=root, download=True, transform=train_trans)
        _test_dataset = datasets.FashionMNIST(root=root, download=True, train=False, transform=test_trans)

        # create iterators
        tr_batch_size = self.data_config.get('train_batch_size', 1)
        test_batch_size = self.data_config.get('test_batch_size', 512)
        train_loader = DataLoader(dataset=_train_dataset, batch_size=tr_batch_size, shuffle=True)
        print('Num of Batches in Train Loader = {}'.format(len(train_loader)))
        test_loader = DataLoader(dataset=_test_dataset, batch_size=test_batch_size)

        return train_loader, test_loader


# class ExtendedMNIST(VisionDataManager):
#     def __init__(self):
#         VisionDataManager.__init__(self)
#
#     def download_data(self, seed=1):
#         torch.manual_seed(seed)
#         _train_dataset = datasets.EMNIST(root=root, download=True, split='balanced')
#
#         # Whiten the Data
#         mean, std = self._get_common_data_trans(_train_dataset)
#         train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
#         test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
#         _train_dataset = datasets.EMNIST(root=root, download=True, transform=train_trans, split='balanced')
#         _test_dataset = datasets.EMNIST(root=root, download=True, train=False, transform=test_trans, split='balanced')
#
#         return _train_dataset, _test_dataset


# class CIFAR10(VisionDataManager):
#     def __init__(self):
#         VisionDataManager.__init__(self)
#
#     def download_data(self):
#         _train_dataset = datasets.CIFAR10(root=root, download=True)
#
#         # Whiten the Data
#         mean, std = self._get_common_data_trans(_train_dataset)
#         train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
#         test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
#         _train_dataset = datasets.CIFAR10(root=root, download=True, transform=train_trans)
#         _test_dataset = datasets.CIFAR10(root=root, download=True, train=False, transform=test_trans)
#
#         return _train_dataset, _test_dataset
#
#
# class ImageNet(VisionDataManager):
#     def __init__(self):
#         VisionDataManager.__init__(self)
#
#     # noinspection PyTypeChecker
#     def download_data(self):
#         _train_dataset = datasets.ImageNet(root=root, download=True)
#
#         # Whiten the Data
#         mean, std = self._get_common_data_trans(_train_dataset)
#         train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
#         test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
#         _train_dataset = datasets.ImageNet(root=root, download=True, transform=train_trans)
#         _test_dataset = datasets.ImageNet(root=root, download=True, train=False, transform=test_trans)
#
#         return _train_dataset, _test_dataset
