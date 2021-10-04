from typing import Dict
import numpy as np
import torch
import time
import math
# from src.attack_manager import get_feature_attack, get_grad_attack
from src.model_manager import (get_model,
                               get_loss,
                               get_scheduler,
                               get_optimizer)
from src.data_manager import process_data
from src.aggregation_manager import get_gar
from src.compression_manager import get_jac_compression_operator


class TrainPipeline:
    def __init__(self, config, seed):
        # ------------------------ Fetch configs ----------------------- #
        print('---- Fetching configs and Initializing stuff -----')
        self.config = config

        self.data_config = config["data_config"]
        self.training_config = config["training_config"]

        self.train_batch_size = self.data_config.get('train_batch_size')
        self.num_batches = self.data_config.get('num_clients', 1)

        self.num_epochs = self.training_config.get('global_epochs', 10)
        self.eval_freq = self.training_config.get('eval_freq', 10)

        self.learner_config = self.training_config["learner_config"]
        self.optimizer_config = self.training_config.get("optimizer_config", {})

        self.lrs_config = self.optimizer_config.get('lrs_config')
        self.loss_fn = self.optimizer_config.get('loss', 'ce')

        self.aggregation_config = self.training_config["aggregation_config"]
        self.jac_compression_config = self.aggregation_config.get("jacobian_compression_config", {})

        self.grad_attack_config = self.aggregation_config.get("grad_attack_config", {})
        self.feature_attack_config = self.data_config.get("feature_attack_config", {})

        # ------------------------ initializations ----------------------- #
        self.metrics = self.init_metric()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.net = self.learner_config.get("net", 'lenet')

        self.epoch = 0
        data_manager = process_data(data_config=self.data_config)
        self.train_loader, self.test_loader = data_manager.get_data_iterator()

        self.model = get_model(learner_config=self.learner_config,
                               data_config=self.data_config,
                               seed=seed,
                               additional_conf=data_manager.additional_model_conf)
        self.criterion = get_loss(loss_fn=self.loss_fn)
        self.optimizer = get_optimizer(params=self.model.parameters(),
                                              optimizer_config=self.optimizer_config)
        self.lrs = get_scheduler(optimizer=self.optimizer,
                                 lrs_config=self.lrs_config)

        # # Compression Operator
        self.C_J = get_jac_compression_operator(jac_compression_config=self.jac_compression_config)
        self.I_k = None  # indices when sparse approx Jac to run aggregation faster

        self.gar = get_gar(aggregation_config=self.aggregation_config)
        self.G = None

        # # for adversarial - get attack model
        # self.feature_attack_model = get_feature_attack(attack_config=self.feature_attack_config)
        # self.grad_attack_model = get_grad_attack(attack_config=self.grad_attack_config)

    def apply_gar(self):
        raise NotImplementedError

    def evaluate_classifier(self,
                            epoch: int,
                            num_epochs: int,
                            model,
                            train_loader,
                            test_loader,
                            metrics,
                            device="cpu") -> float:
        train_error, train_acc, train_loss = self._evaluate(model=model, data_loader=train_loader, device=device)
        test_error, test_acc, _ = self._evaluate(model=model, data_loader=test_loader, device=device)

        if test_acc > metrics["best_test_acc"]:
            metrics["best_test_acc"] = test_acc
        print('Epoch progress: {}/{}, train loss = {}, train acc = {}, test acc = {}, best acc ={}'.
              format(epoch, num_epochs, train_loss, train_acc, test_acc, metrics["best_test_acc"]))

        metrics["test_error"].append(test_error)
        metrics["test_acc"].append(test_acc)
        metrics["train_error"].append(train_error)
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)

        return train_loss

    def _evaluate(self, model, data_loader, verbose=False, device="cpu"):
        model.to(device)
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            batches = 0

            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                # if criterion is not None:
                loss = self.criterion(outputs, labels, evaluate=True)
                total_loss += loss.item()

                batches += 1
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            total_loss /= batches
            if verbose:
                print('Accuracy: {} %'.format(acc))
            return 100 - acc, acc, total_loss

    def init_metric(self):
        metrics = {"config": self.config,

                   "num_param": 0,

                   # Train and Test Performance
                   "train_loss": [],
                   "test_error": [],
                   "test_acc": [],
                   "best_test_acc": 0,

                   # Compression related residuals
                   "jacobian_residual": [],

                   # compute Time stats per epoch
                   "epoch_compression_cost": [],
                   "epoch_grad_cost": [],
                   "epoch_agg_cost": [],
                   "epoch_gm_iter": [],

                   # Total Costs
                   "total_cost": 0,
                   "total_grad_cost": 0,
                   "total_agg_cost": 0,
                   "total_compression_cost": 0,

                   "total_gm_iter": 0,
                   "avg_gm_cost": 0,

                   "num_iter": 0,
                   "num_opt_steps": 0,
                   "num_of_communication": 0,
                   "num_grad_steps": 0,
                   }
        return metrics

    def run_train(self):
        raise NotImplementedError("This method needs to be implemented for each pipeline")

    def run_batch_train(self):
        raise NotImplementedError("This method needs to be implemented for each pipeline")

    def run_fed_train(self):
        raise NotImplementedError("This method needs to be implemented for each pipeline")
