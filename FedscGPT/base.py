import torch.nn as nn
import torch
import time
import os
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.model import TransformerModel, AdversarialDiscriminator
from FedscGPT.utils import load_config, load_fed_config, get_logger, weighted_average, average_weights, get_cuda_device, \
    split_data_by_batch, save_data_batches

class BaseMixin:
    """
    Base class for scGPT tasks.

    """
    def __init__(self, task, config_file, celltype_key, batch_key, output_dir, logger=None,
                 log_id: [str or None] = None, gpu: int = 0, verbose: bool = False, n_epochs=None, mu=None, use_fedprox=False, init_weights_dir=None, **kwargs):
        self.log_id = log_id
        self.log_level = "info" if log_id is None else log_id
        self.output_dir = output_dir
        self.celltype_key = celltype_key
        self.batch_key = batch_key
        self.config = load_config(config_file, task, verbose)
        if n_epochs:
            self.config.train.epochs = n_epochs
        self.logger = get_logger(output_dir) if logger is None else logger
        self.loss_meter = LossMeter(**self.config.train.__dict__)
        self.optimizers = {}
        self.losses = {}
        self.lr_schedulers = {}
        self.discriminator = None
        self.losses = {}
        self.loss_args = {}
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.train.amp)
        self.model = None
        self.device = get_cuda_device(gpu)
        self.verbose = verbose
        self.mu = mu
        self.use_fedprox = use_fedprox
        self.global_model = None
        self.init_weights_dir = init_weights_dir

    def log(self, msg, log_level=None):
        if log_level is None:
            log_level = self.log_level
        getattr(self.logger, log_level)(msg)

    def sanity_check(self):
        assert self.config.preprocess.input_style in ["normed_raw", "log1p", "binned"], \
            f"input_style {self.config.preprocess.input_style} not supported."
        assert self.config.preprocess.output_style in ["normed_raw", "log1p", "binned"], \
            f"output_style {self.config.preprocess.output_style} not supported."
        assert self.config.preprocess.input_emb_style in ["category", "continuous", "scaling"], \
            f"input_emb_style {self.config.preprocess.input_emb_style} not supported."
        if self.config.preprocess.input_style == "binned":
            if self.config.preprocess.input_emb_style == "scaling":
                raise ValueError("input_emb_style `scaling` is not supported for binned input.")
        elif self.config.preprocess.input_style == "log1p" or self.config.preprocess.input_style == "normed_raw":
            if self.config.preprocess.input_emb_style == "category":
                raise ValueError(
                    "input_emb_style `category` is not supported for log1p or normed_raw input."
                )
        if self.config.train.ADV and self.config.train.DAB:
            raise ValueError("ADV and DAB cannot be both True.")

    def set_DAB(self):
        self.losses["dab"] = nn.CrossEntropyLoss()
        self.loss_args["dab"] = ["dab_output", 'batch_labels']
        self.optimizers['dab'] = torch.optim.Adam(self.model.parameters(), lr=self.config.train.lr)
        self.lr_schedulers['dab'] = torch.optim.lr_scheduler.StepLR(
            self.optimizers['dab'], self.config.train.schedule_interval, gamma=self.config.train.schedule_ratio
        )

    def set_defualt(self):
        self.optimizers['main'] = torch.optim.Adam(
            self.model.parameters(), lr=self.config.train.lr, eps=1e-4 if self.config.train.amp else 1e-8
        )
        self.lr_schedulers['main'] = torch.optim.lr_scheduler.StepLR(
            self.optimizers['main'], self.config.train.schedule_interval, gamma=self.config.train.schedule_ratio
        )

    def set_mse(self):
        self.losses["mse"] = masked_mse_loss
        self.loss_args["mse"] = ["mlm_output", 'target_values', 'masked_positions']

    def set_cls(self):
        self.losses["cls"] = nn.CrossEntropyLoss()
        self.loss_args["cls"] = ["cls_output", 'celltype_labels']

    def set_ADV(self):
        self.losses["adv"] = nn.CrossEntropyLoss()  # consider using label smoothing
        self.loss_args["adv"] = ["adv_output", 'batch_labels']
        self.optimizers["E"] = torch.optim.Adam(self.model.parameters(), lr=self.config.train.ADV.lr)
        self.lr_schedulers["E"] = torch.optim.lr_scheduler.StepLR(
            self.optimizers['E'], self.config.train.schedule_interval, gamma=self.config.train.schedule_ratio
        )
        self.optimizers["D"] = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.train.ADV.lr)
        self.lr_schedulers["D"] = torch.optim.lr_scheduler.StepLR(
            self.optimizers['D'], self.config.train.schedule_interval, gamma=self.config.train.schedule_ratio
        )

    def set_zero_prob(self):
        self.losses["zero_log_prob"] = criterion_neg_log_bernoulli
        self.loss_args["zero_log_prob"] = ["mlm_zero_probs", 'target_values', 'masked_positions']

    def set_cce(self):
        self.losses["cce"] = lambda x: 10 * x
        self.loss_args["cce"] = ['loss_cce']

    def set_mvc(self):
        self.losses["mvc"] = masked_mse_loss
        self.loss_args["mvc"] = ["mvc_output", 'target_values', 'masked_positions']

    def set_ecs(self):
        self.losses["ecs"] = lambda x: 10 * x
        self.loss_args["ecs"] = ['loss_ecs']

    def set_mvc_zero_log_prob(self):
        self.losses["mvc_zero_log_prob"] = criterion_neg_log_bernoulli
        self.loss_args["mvc_zero_log_prob"] = ["mvc_zero_probs", 'target_values', 'masked_positions']

    def instantiate_adv(self):
        self.discriminator = AdversarialDiscriminator(
            d_model=self.config.model.embsize,
            n_cls=self.config.model.num_batch_types,
        ).to(self.device)

    def setup_losses(self):
        self.set_defualt()
        if self.config.train.MLM:
            self.set_mse()
        if self.config.train.CLS:
            self.set_cls()
        if self.config.train.DAB:
            self.set_DAB()
        if self.config.train.ADV:
            self.instantiate_adv()
            self.set_ADV()
        if self.config.model.explicit_zero_prob:
            self.set_zero_prob()
        if self.config.train.MVC and self.config.model.explicit_zero_prob:
            self.set_mvc_zero_log_prob()
        if self.config.train.MVC:
            self.set_mvc()
        if self.config.train.ECS:
            self.set_ecs()
        if self.config.train.CCE:
            self.set_cce()

    def apply_loss(self, **kwargs):
        self.loss_meter.update_count(kwargs["target_values"].size(0))
        for loss_name in self.losses.keys():
            loss_args = [kwargs[arg] for arg in self.loss_args[loss_name]]
            loss = self.losses[loss_name](*loss_args)
            self.loss_meter.update(loss_name, loss)

        if 'cls' in self.losses:
            error_rate = 1 - ((kwargs["cls_output"].argmax(1) == kwargs['celltype_labels']).sum().item()
                              ) / kwargs['celltype_labels'].size(0)
            self.loss_meter.update_error(error_rate)

    def lr_schedulers_step(self):
        for scheduler in self.lr_schedulers.values():
            scheduler.step()

    def save_best_model(self, model=None):
        if model is None:
            model = self.best_model
        torch.save(model.state_dict(), f"{self.output_dir}/model/model.pt")

    def freeze_params(self):
        pre_freeze_param_count = sum(
            dict((p.data_ptr(), p.numel()) for p in self.model.parameters() if p.requires_grad).values())
        # Freeze all pre-decoder weights
        for name, para in self.model.named_parameters():
            if self.verbose:
                print("-" * 20)
                print(f"name: {name}")
            if self.config.train.freeze and "encoder" in name and "transformer_encoder" not in name:
                # if config.freeze and "encoder" in name:
                if self.verbose:
                    print(f"freezing weights for: {name}")
                para.requires_grad = False
        post_freeze_param_count = sum(
            dict((p.data_ptr(), p.numel()) for p in self.model.parameters() if p.requires_grad).values())
        self.log(f"Total Pre freeze Params {(pre_freeze_param_count)}")
        self.log(f"Total Post freeze Params {(post_freeze_param_count)}")

    def load_matched_param(self, model_dir):
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(model_dir)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        if self.verbose:
            for k, v in pretrained_dict.items():
                self.log(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def load_init_weights(self):
        if os.path.exists(self.init_weights_dir):
            if self.verbose:
                self.log(f"Loading the initial model weights from {self.init_weights_dir}")
            self.model.load_state_dict(torch.load(self.init_weights_dir))
            return False
        if self.verbose:
            self.log(f"Initial model weights not found at {self.init_weights_dir} thw weights will be stored there...")
        return True


    def save_init_weights(self):
        torch.save(self.model.state_dict(), self.init_weights_dir)

    def move_to_gpu(self):
        model_device = next(self.model.parameters()).device
        if model_device != self.device:
            self.model.to(self.device)
        else:
            self.log(f"Model is already on {self.device} device, no need to move it.")

        if self.discriminator is not None:
            disc_device = next(self.discriminator.parameters()).device
            if disc_device != self.device:
                self.discriminator.to(self.device)
            else:
                self.log(f"Discriminator is already on {self.device} device, no need to move it.")

    def move_to_cpu(self):
        model_device = next(self.model.parameters()).device
        if model_device != torch.device("cpu"):
            self.model.to("cpu")
        else:
            self.log(f"Model is already on CPU device, no need to move it.")

        if self.discriminator is not None:
            disc_device = next(self.discriminator.parameters()).device
            if disc_device != torch.device("cpu"):
                self.discriminator.to("cpu")
            else:
                self.log(f"Discriminator is already on CPU device, no need to move it.")


class LossMeter:
    def __init__(self, MLM, CLS, CCE=False, MVC=False, ECS=False, DAB=False, ADV=False, explicit_zero_prob=False,
                 **kwargs):
        self.loss: torch.tensor = 0.0
        self.batch_loss: torch.tensor = 0.0
        self.error = 0.0
        self.count = 0
        self.num_batches = 0
        self.loss_dict = {}
        self.start_time = time.time()
        if MLM:
            self.loss_dict["mse"] = 0.0
        if CLS:
            self.loss_dict["cls"] = 0.0
        if CCE:
            self.loss_dict["cce"] = 0.0
        if MVC:
            self.loss_dict["mvc"] = 0.0
        if ECS:
            self.loss_dict["ecs"] = 0.0
        if DAB:
            self.loss_dict["dab"] = 0.0
            assert "dab_weight" in kwargs, "DAB weight is not provided"
            self.dab_weight = kwargs["dab_weight"]
        if ADV:
            self.loss_dict["adv_e"] = 0.0
            self.loss_dict["adv_d"] = 0.0
        if explicit_zero_prob:
            self.loss_dict["zero_log_prob"] = 0.0
        if MVC and explicit_zero_prob:
            self.loss_dict["mvc_zero_log_prob"] = 0.0

    def update(self, loss_name, loss: torch.tensor):
        self.loss_dict[loss_name] += loss
        if loss_name == "dab":
            loss = loss * self.dab_weight
        self.batch_loss += loss

    def reset_batch_loss(self):
        self.loss += self.batch_loss
        self.batch_loss = 0.0

    def update_count(self, count):
        self.count += count
        self.num_batches += 1

    def get_loss(self, loss_name):
        return self.loss_dict[loss_name]

    def update_error(self, error):
        self.error += error

    def reset(self):
        self.start_time = time.time()
        self.loss = 0.0
        self.batch_loss = 0.0
        self.error = 0.0
        self.num_batches = 0
        for k in self.loss_dict:
            self.loss_dict[k] = 0.0

    def log(self, log_interval):
        ms_per_batch = (time.time() - self.start_time) * 1000 / log_interval
        log_txt = f"ms/batch {ms_per_batch:5.2f} | loss {(self.loss.item() / log_interval):5.2f} | "
        for loss_name in self.loss_dict:
            log_txt += f"{loss_name} {(self.loss_dict[loss_name].item() / log_interval):5.2f} |"
        log_txt += f"Error {(self.error / log_interval):5.2f}"
        return log_txt


class FedBase:

    def __init__(self, fed_config_file, task, data_dir, output_dir, client_ids=None, n_rounds=None, smpc=False, **kwargs):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.fed_config = load_fed_config(fed_config_file, task)
        if "gpu" in kwargs:
            self.device = get_cuda_device(kwargs["gpu"])
        if n_rounds:
            self.fed_config.n_rounds = n_rounds
        self.client_ids = client_ids
        self.smpc = smpc
        self.logger = None
        self.aggregation_type = self.fed_config.aggregation_type
        self.init_model = None
        self.global_weights = None
        self.global_model_keys = None
        self.global_weight_shapes = None
        self.global_model = None
        self.local_model_weights = []
        self.local_n_samples = []
        self.n_total_samples = 0
        self.clients = []
        self.n_clients = None
        self.n_rounds = self.fed_config.n_rounds
        self.round = 0
        self.clients_data_dir = []
        self.clients_output_dir = []

    def get_n_local_samples(self):
        self.local_n_samples = [client.n_samples for client in self.clients]
        self.n_total_samples = sum(self.local_n_samples)

    def train(self):
        while True:
            self.round += 1
            for client in self.clients:
                client.train()
            self.collect_local_weights()
            self.global_aggregate()
            self.update_clients_model()
            if self.stopping_criterion:
                break

    def stopping_criterion(self):
        return self.round > self.n_rounds

    def global_aggregate(self):
        if self.aggregation_type == "average":
            return average_weights(self.local_model_weights)
        elif self.aggregation_type == "weighted_average":
            return weighted_average(self.local_model_weights, self.local_n_samples)
        else:
            raise ValueError(f"{self.aggregation_type} is not supported as an aggregation type")

    def collect_local_weights(self):
        self.local_model_weights = []
        for client in self.clients:
            self.local_model_weights.append(client.model.state_dict())

    def update_clients_model(self, **kwargs):
        return [client.local_update(self.global_weights, **kwargs) for client in self.clients]

    def create_dirs(self, path):

        if not os.path.exists(path):
            raise NotADirectoryError(f"{path} does not exist!")
        dirs = []
        for c in range(self.n_clients):
            client_dir = os.path.join(path, f"client_{c}")
            if not os.path.exists(client_dir):
                os.makedirs(client_dir)
            dirs.append(client_dir)
        return dirs

    def distribute_adata_by_batch(self, adata, batch_key, keep_vars=False):
        """
        Dynamic distribution of adata by batch id.
        Parameters
        ----------
        adata

        Returns
        -------

        """
        filename = "adata.h5ad"
        self.n_clients = len(adata.obs[batch_key].unique())
        self.client_ids = sorted(adata.obs[batch_key].unique()) if self.client_ids is None else self.client_ids
        self.logger = get_logger(self.output_dir, "FedscGPT", self.client_ids)
        self.clients_data_dir = [f"{self.data_dir}/client_{batch}" for batch in self.client_ids]
        if not all([os.path.exists(f"{d}/{filename}") for d in self.clients_data_dir]):
            batches = split_data_by_batch(adata, batch_key, keep_vars)
            for dir in self.clients_data_dir:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            assert self.n_clients == len(self.clients_data_dir), \
                f"Number of clients directories {len(self.clients_data_dir)} does not match the number of clients {self.n_clients}"
            save_data_batches(batches, self.clients_data_dir, filename, keep_vars)
        self.clients_output_dir = self.create_dirs(self.output_dir)

    
    def save_clients_fed_prep_data(self):
        for client in self.clients:
            client.adata.write_h5ad(f"{self.clients_data_dir[client]}/fed_prep_adata.h5ad")
    
    def retain_best_model(self, on=True):
        self.logger.federated(f"Retain best model: {on}")
        for client in self.clients:
            client.config.log.retain_best_model = on


    def init_global_weights(self):
        self.global_weights = torch.load(self.clients[0].init_weights_dir)
        self.global_model_keys = list(self.global_weights.keys())
        self.global_weight_shapes = {key: tensor.shape for key, tensor in self.global_weights.items()}