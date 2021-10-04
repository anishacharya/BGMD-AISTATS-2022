import torch
from torch.utils.data import DataLoader
import argparse
import json
import os
import yaml
import numpy as np
from numpyencoder import NumpyEncoder

from src.training_manager import TrainPipeline
from src.data_manager import process_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class ActiveSamplingRobust(TrainPipeline):
    """
        This Pipeline implements robust SGD + various online active sample selection methods
    """
    def __init__(self, config, seed):
        TrainPipeline.__init__(self, config=config, seed=seed)
        self.epoch = 0
        data_manager = process_data(data_config=self.data_config)
        self.train_loader, self.test_loader = data_manager.download_data()

    def run_train(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def run_batch_train(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        while self.epoch < self.num_epochs:
            self.model.to(device)
            self.model.train()

    def run_fed_train(self):
        raise NotImplementedError("This method needs to be implemented for each pipeline")


def _parse_args():
    parser = argparse.ArgumentParser(description='federated/decentralized/distributed training experiment template')
    parser.add_argument('--train_mode',
                        type=str,
                        default='vanilla',
                        help='vanilla: launch regular batch sgd'
                             'distributed: launch distributed Training '
                             'fed: launch federated training')
    parser.add_argument('--conf',
                        type=str,
                        default=None,
                        help='Pass Config file path')
    parser.add_argument('--o',
                        type=str,
                        default='default_output',
                        help='Pass result file path')
    parser.add_argument('--dir',
                        type=str,
                        default=None,
                        help='Pass result file dir')
    parser.add_argument('--n_repeat',
                        type=int,
                        default=1,
                        help='Specify number of repeat runs')
    args = parser.parse_args()
    return args


def run_main():
    args = _parse_args()
    print(args)
    root = os.getcwd()
    config_path = args.conf if args.conf else root + '/configs/default_robust_config.yaml'
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    # Training - Repeat over the random seeds #
    # ----------------------------------------
    results = []
    for seed in np.arange(args.n_repeat):
        # ----- Launch Training ------ #
        train_mode = args.train_mode
        trainer = ActiveSamplingRobust(config=config, seed=seed)

        if train_mode == 'vanilla':
            # Launch Vanilla mini-batch Training
            trainer.run_train(config=config, seed=seed)
            results.append(trainer.metrics)
        else:
            raise NotImplementedError

    # Write Results #
    # ----------------
    directory = args.dir if args.dir else "result_dumps/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + args.o, 'w+') as f:
        json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


if __name__ == '__main__':
    run_main()
