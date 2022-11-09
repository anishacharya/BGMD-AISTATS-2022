import torch
import argparse
import json
import os
import yaml
import numpy as np
from numpyencoder import NumpyEncoder
import random


def _parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--conf', type=str, default='configs/cifar10_config.yaml', help='Pass Config file path')
    parser.add_argument('--log_path', type=str, default='./default_log', help='Pass result file path')
    parser.add_argument('--op_model_path', type=str, default='./default_model',
                        help='model to save - Provide Full Path')
    parser.add_argument('--ip_model_path', type=str, default=None, help='starting model path - if None then rand init')
    # parser.add_argument('--train_mode', type=str, default='finetune', help='encode/finetune/linear_probe')
    parser.add_argument('--n_repeat', type=int, default=1, help='Specify number of repeat runs')
    args = parser.parse_args()
    return args


def run_main():
    args = _parse_args()
    print(args)
    config = yaml.load(open(args.conf), Loader=yaml.FullLoader)
    # create log and directory if it does not exist
    if not os.path.exists(os.path.split(args.log_path)[0]):
        os.makedirs(os.path.split(args.log_path)[0])
    if not os.path.exists(os.path.split(args.op_model_path)[0]):
        os.makedirs(os.path.split(args.op_model_path)[0])
