from typing import Dict
import numpy as np
import torch
import math
# from src.attack_manager import get_feature_attack, get_grad_attack
from src.model_manager import get_model, get_optimizer
#                                get_scheduler,
#                                get_loss)
# from src.aggregation_manager import get_gar
# from src.compression_manager import get_compression_operator, get_jac_compression_operator


class TrainPipeline:
    def __init__(self, config, seed):
        # ------------------------ Fetch configs ----------------------- #
        print('---- Fetching configs and Initializing stuff -----')
        self.config = config

        self.data_config = config["data_config"]
        self.training_config = config["training_config"]

        self.train_batch_size = self.data_config.get('train_batch_size')
        # self.num_batches = self.training_config.get('num_clients', 1)
        self.num_epochs = self.training_config.get('global_epochs', 10)
        self.eval_freq = self.training_config.get('eval_freq', 10)

        self.learner_config = self.training_config["learner_config"]
        self.optimizer_config = self.training_config.get("optimizer_config", {})
        self.client_optimizer_config = self.optimizer_config.get("client_optimizer_config", {})
        self.client_lrs_config = self.optimizer_config.get('client_lrs_config')

        # self.criterion = get_loss(loss=self.client_optimizer_config.get('loss', 'ce'))
        self.sampler = self.client_optimizer_config.get('loss_sampling', None)
        self.beta_loss = self.client_optimizer_config.get('initial_loss_sampling_fraction', 1)

        # self.aggregation_config = self.training_config["aggregation_config"]
        #
        # self.grad_compression_config = self.aggregation_config.get("gradient_compression_config", {})
        # self.jac_compression_config = self.aggregation_config.get("jacobian_compression_config", {})

        # self.grad_attack_config = self.aggregation_config.get("grad_attack_config", {})
        # self.feature_attack_config = self.data_config.get("feature_attack_config", {})

        # ------------------------ initializations ----------------------- #
        self.metrics = self.init_metric()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.net = self.learner_config.get("net", 'lenet')
        self.model = get_model(learner_config=self.learner_config,
                               seed=seed)
        self.client_optimizer = get_optimizer(params=self.model.parameters(),
                                              optimizer_config=self.client_optimizer_config)
        # self.client_lrs = get_scheduler(optimizer=self.client_optimizer,
        #                                 lrs_config=self.client_lrs_config)
        #
        # # Compression Operator
        # self.C_J = get_jac_compression_operator(jac_compression_config=self.jac_compression_config)
        # self.I_k = None  # indices when sparse approx Jac to run aggregation faster
        #
        # self.C_g = get_compression_operator(compression_config=self.grad_compression_config)
        #
        # self.gar = get_gar(aggregation_config=self.aggregation_config)
        # self.G = None
        #
        # # for adversarial - get attack model
        # self.feature_attack_model = get_feature_attack(attack_config=self.feature_attack_config)
        # self.grad_attack_model = get_grad_attack(attack_config=self.grad_attack_config)

    def init_metric(self):
        metrics = {"config": self.config,

                   "num_param": 0,

                   # Train and Test Performance
                   "train_loss": [],
                   "fine_tune_loss": [],
                   "test_error": [],
                   "test_acc": [],
                   "best_test_acc": 0,

                   # Compression related residuals
                   "gradient_residual": [],
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