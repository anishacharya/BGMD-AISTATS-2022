import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
from typing import Dict
import random

SEED = 1234
MAX_VOCAB_SIZE = 25_000

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy',
                  tokenizer_language='en_core_web_sm',
                  include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)


class NLPDataManager:
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


class IMDB(NLPDataManager):
    def __init__(self, data_config: Dict, transformation=None):
        NLPDataManager.__init__(data_config=data_config, transformation=transformation)

    def download_data(self):
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        TEXT.build_vocab(train_data,
                         max_size=MAX_VOCAB_SIZE,
                         vectors="glove.6B.100d",
                         unk_init=torch.Tensor.normal_)
        LABEL.build_vocab(train_data)

