import torch
from torchtext.legacy import data, datasets
from typing import Dict

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

    def __init__(self, data_config: Dict):
        self.data_config = data_config
        self.tr_batch_size = self.data_config.get('train_batch_size', 1)
        self.test_batch_size = self.data_config.get('test_batch_size', 512)

    def get_data_iterator(self):
        """ Downloads Data and Apply appropriate Transformations . returns train, test dataset """
        raise NotImplementedError("This method needs to be implemented")


class IMDB(NLPDataManager):
    def __init__(self, data_config: Dict):
        NLPDataManager.__init__(self, data_config=data_config)

    def get_data_iterator(self):
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        TEXT.build_vocab(train_data,
                         max_size=MAX_VOCAB_SIZE,
                         vectors="glove.6B.100d",
                         unk_init=torch.Tensor.normal_)
        LABEL.build_vocab(train_data)

        train_loader = data.BucketIterator.splits(train_data, batch_size=self.tr_batch_size)
        test_loader = data.BucketIterator.splits(test_data, batch_size=self.test_batch_size)

        return train_loader, test_loader


