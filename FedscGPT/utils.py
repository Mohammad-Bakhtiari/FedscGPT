"""
# This code includes the 'compute_perturbation_metrics' method sourced from:
# https://github.com/bowang-lab/scGPT/blob/7301b51a72f5db321fccebb51bc4dd1380d99023/scgpt/utils/util.py
# Original repository by Bo Wang Lab, available under the respective license.

"""
import random
import numpy as np
import torch
import tensorflow as tf
from anndata import AnnData
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    tf.random.set_seed(seed)
set_seed()
from matplotlib import colors
import os
import anndata
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import scgpt as scg
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Any
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
import dataclasses
import yaml
import scanpy as sc
import pickle
import matplotlib.pyplot as plt
import logging
import sys



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed + SEED)
    random.seed(worker_seed + SEED)
    torch.manual_seed(worker_seed + SEED)

BASE_CLIENT_LEVEL_NUM = 35


@dataclasses.dataclass
class ADVTrainConfig:
    E_delay_epochs: int
    D_delay_epochs: int
    lr: float


@dataclasses.dataclass
class TrainConfig:
    dab_weight: float
    lr: float
    batch_size: int
    eval_batch_size: int
    epochs: int
    schedule_ratio: float
    schedule_interval: int
    amp: bool
    save_eval_interval: int
    MLM: bool
    CLS: bool
    ADV: bool or ADVTrainConfig
    CCE: bool
    MVC: bool
    ecs_thres: float
    DAB: bool
    INPUT_BATCH_LABELS: bool
    mvc_decoder_style: str
    freeze: bool
    do_sample_in_train: bool
    DSBN: bool
    ECS: bool = dataclasses.field(init=False)
    early_stop: int = None
    load_param_prefixs: list = None

    def __post_init__(self):
        self.ECS = self.ecs_thres > 0


@dataclasses.dataclass
class ModelConfig:
    embsize: int
    nhead: int
    d_hid: int
    nlayers: int
    nlayers_cls: int
    n_cls: int
    dropout: float
    do_mvc: bool
    do_dab: bool
    use_batch_labels: bool
    domain_spec_batchnorm: bool
    input_emb_style: str
    n_input_bins: int
    cell_emb_style: str
    mvc_decoder_style: str
    ecs_threshold: float
    explicit_zero_prob: bool
    use_fast_transformer: bool
    fast_transformer_backend: str
    pre_norm: bool


@dataclasses.dataclass
class DatasetConfig:
    raw_data_key: str = "X"
    data_is_raw: bool = False
    filter_gene_by_counts: bool = False
    filter_cell_by_counts: bool = False
    normalize_total: int = 1e4  # 3. whether to normalize the raw data and to what sum
    result_normed_key: str = "X_normed"  # the key in adata.layers to store the normalized data
    log1p: bool = False  # 4. whether to log1p the normalized data
    result_log1p_key: str = "X_log1p"
    subset_hvg: bool = False  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor: str = "cell_ranger"
    result_binned_key: str = "X_binned"

    def __post_init__(self):
        if isinstance(self.normalize_total, str):
            self.normalize_total = float(self.normalize_total)


@dataclasses.dataclass
class PreprocessConfig:
    n_bins: int
    pre_norm: bool
    include_zero_gene: bool
    input_style: str
    output_style: str
    input_emb_style: str
    mask_ratio: float
    cell_emb_style: str
    pad_token: str
    special_tokens: list
    mask_value: str or int
    max_seq_len: int
    per_seq_batch_sample: bool
    pad_value: int = None
    pert_pad_id: int = None


@dataclasses.dataclass
class LogConfig:
    log_interval: int
    save_eval_interval: int
    do_eval_scib_metrics: bool
    retain_best_model: bool
    log_error: bool = False
    pool_size: int = None
    top_k: int = 15


@dataclasses.dataclass
class Config:
    preprocess: PreprocessConfig
    train: TrainConfig
    model: ModelConfig
    dataset: DatasetConfig
    log: LogConfig


@dataclasses.dataclass
class FedPreprocessConfig:
    filter_gene_by_counts: bool
    filter_cell_by_counts: bool
    normalize_total: bool
    log1p: bool
    subset_hvg: bool
    binning: bool


@dataclasses.dataclass
class FedConfig:
    n_rounds: int
    aggregation_type: str
    condition_key: str
    preprocess: FedPreprocessConfig


def load_config(file_path: str, task, verbose) -> Config:
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    if verbose:
        print_config(config_dict)
    preprocess_config = PreprocessConfig(**config_dict[task]['preprocess'])
    train_config = TrainConfig(**config_dict[task]['train'])
    model_config = ModelConfig(**config_dict[task]['model'])
    log_config = LogConfig(**config_dict[task]['log'])
    dataset_config = DatasetConfig(**config_dict[task]['dataset'])
    return Config(preprocess=preprocess_config,
                  train=train_config,
                  model=model_config,
                  dataset=dataset_config,
                  log=log_config
                  )


def load_fed_config(file_path: str, task: str) -> FedConfig:
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    print("Federated Config")
    print_config(config_dict)
    preprocess_config = FedPreprocessConfig(**config_dict[task]['preprocess'])
    config_dict[task].pop('preprocess')
    return FedConfig(**config_dict[task], preprocess=preprocess_config)


def print_config(config: dict or tuple, level=0):
    for k, v in config.items():
        if isinstance(v, dict):
            print("  " * level + str(k) + ":")
            print_config(v, level + 1)
        else:
            print("  " * level + str(k) + ":", v)


def get_cuda_device(device_index: int):
    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)  # Set the device globally
        return torch.device(f"cuda:{device_index}")
    else:
        return torch.device("cpu")


def read_h5ad(data_dir, adata):
    if os.path.isabs(adata):
        print(f"Reading data from {adata} ...")
        adata = anndata.read_h5ad(adata)
    else:
        print(f"Reading data from {data_dir}/{adata} ...")
        adata = anndata.read_h5ad(f"{data_dir}/{adata}")
    return adata


def add_federated_logging(logger):
    FEDERATED_LEVEL_NUM = 25
    logging.addLevelName(FEDERATED_LEVEL_NUM, "FEDERATED")

    def federated(self, message, *args, **kws):
        if self.isEnabledFor(FEDERATED_LEVEL_NUM):
            self._log(FEDERATED_LEVEL_NUM, message, args, **kws)

    setattr(logging.Logger, 'federated', federated)
    logger.setLevel(min(logger.level, FEDERATED_LEVEL_NUM))  # Ensure logger level includes the new custom level


def add_inference_logging(logger):
    INFERENCE_LEVEL_NUM = 26  # Using a different level number to avoid conflict
    logging.addLevelName(INFERENCE_LEVEL_NUM, "INFERENCE")

    def inference(self, message, *args, **kws):
        if self.isEnabledFor(INFERENCE_LEVEL_NUM):
            self._log(INFERENCE_LEVEL_NUM, message, args, **kws)

    setattr(logging.Logger, 'inference', inference)
    logger.setLevel(min(logger.level, INFERENCE_LEVEL_NUM))  # Ensure logger level includes the new custom level


def add_client_logging(logger, client_id, level_num):
    level_name = f"CLIENT_{client_id}"
    logging.addLevelName(level_num, level_name)

    def log_for_client(self, message, *args, **kws):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kws)

    setattr(logging.Logger, level_name.lower(), log_for_client)
    logger.setLevel(min(logger.level, level_num))  # Ensure logger level includes the new custom level


def get_logger(output_dir, logger_title="scGPT", client_ids=None):
    assert logger_title in ["scGPT", "FedscGPT"], f"Invalid logger title: {logger_title}"
    if client_ids is None:
        client_ids = []

    logger = logging.getLogger(logger_title)
    if not logger.hasHandlers() or len(logger.handlers) == 0:
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        h = logging.FileHandler(f"{output_dir}/run.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        h.setFormatter(formatter)
        h.setLevel(logger.level)
        logger.addHandler(h)

    # Add federated logging level if logger_title is "FedscGPT"
    if logger_title == "FedscGPT":
        add_federated_logging(logger)
        for idx, client_id in enumerate(client_ids):
            add_client_logging(logger, client_id, BASE_CLIENT_LEVEL_NUM + idx)

    add_inference_logging(logger)
    return logger


def per_epoch_data_prep(tokenized_train, tokenized_valid, train_celltype_labels, valid_celltype_labels,
                        train_batch_labels,
                        valid_batch_labels, mask_value, pad_value, mask_ratio, epoch, sort_seq_batch=False):
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader


def get_mixin_config(config):
    return {"lr": config["train"]["lr"],
            "schedule_interval": config["train"]["schedule_interval"],
            "schedule_ratio": config["train"]["schedule_ratio"],
            "amp": config["train"]["amp"],
            }


def model_config_adj(config):
    if config["preprocess"]["input_emb_style"] == "category":
        config['model']['mask_value'] = config["preprocess"]["n_bins"] + 1
        config['model']['pad_value'] = config["preprocess"]["n_bins"]  # for padding gene expr values
        config['model']['n_input_bins'] = config["preprocess"]["n_bins"] + 2
    else:
        config['model']['mask_value'] = -1
        config['model']['pad_value'] = -2
        config['model']['n_input_bins'] = config["preprocess"]["n_bins"]


def plot_umap(adata, cell_type_key, unique_celltypes, file_name, legend='no_legend'):
    # Create a palette for the cell types
    palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    palette_ = palette_ * 3
    palette_ = {c: palette_[i] for i, c in enumerate(unique_celltypes)}

    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 10)  # Create a grid spec with 1 row and 10 columns

    # Use columns 0-4 for the first plot and 4-8 for the second plot
    ax1 = fig.add_subplot(gs[0, 0:5])  # First plot in columns 0-5
    ax2 = fig.add_subplot(gs[0, 5:10])  # Second plot in columns 5-10
    # Plot the UMAP for "celltype" and "predictions"
    for ax, color in zip([ax1, ax2], [cell_type_key, "predictions"]):
        if legend == 'no_legend':
            sc.pl.umap(adata, color=color, palette=palette_, ax=ax, show=False, legend_loc=None)
        elif legend == 'legend_only':
            sc.pl.umap(adata, color=color, palette=palette_, ax=ax, show=False)
        else:
            raise ValueError(f"Invalid value for legend: {legend}")

    if legend == 'no_legend':
        plt.tight_layout()
        plt.savefig(file_name, dpi=300)
    else:
        handles, labels = ax1.get_legend_handles_labels()

        # Plot the legend separately
        fig_legend, ax_legend = plt.subplots(figsize=(9, 3))  # Separate figure for the legend
        ax_legend.legend(handles, labels, loc='center', fontsize='small', frameon=False, ncol=2)
        ax_legend.axis('off')  # Hide the axis for the legend area

        # Save the legend separately
        plt.savefig(file_name, dpi=300)

    plt.close()


def plot(adata, celltype: list, celltype_key: str, save_dir: str):
    if "X_umap" not in adata.obsm.keys() or "X_pca" not in adata.obsm.keys():
        print("UMAP or PCA coordinates are not computed for the dataset. Calculating X_umap for 30 neighbors.")
        sc.pp.neighbors(adata, n_neighbors=30)
    plot_umap(adata, celltype_key, celltype, f"{save_dir}/umap_plots.png", legend='no_legend')
    plot_umap(adata, celltype_key, celltype, f"{save_dir}/legend.png", legend='legend_only')

def dump_results(predictions, labels, results, id2type, save_dir, epoch=None, n_rounds=None):
    save_dict = {
        "predictions": predictions,
        "labels": labels,
        "results": results,
        "id_maps": id2type,
    }
    if epoch is None or n_rounds is None:
        with open(f"{save_dir}/results.pkl", "wb") as f:
            pickle.dump(save_dict, f)
    else:
        save_path = f"{save_dir}/results.pkl"
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                all_results = pickle.load(f)
        else:
            all_results = {}


        # Ensure the dictionary structure exists
        if epoch not in all_results:
            all_results[epoch] = {}

        if n_rounds not in all_results[epoch]:
            all_results[epoch][n_rounds] = {}

        # Store the current result dictionary under the epoch and round
        all_results[epoch][n_rounds] = save_dict

        # Save the updated results back to the file
        with open(save_path, "wb") as f:
            pickle.dump(all_results, f)

        print(f"Results saved for epoch {epoch}, round {n_rounds} in {save_path}")


def eval_annotation(celltypes: list, predictions, labels, id2type, save_dir):
    for i in set([id2type[p] for p in predictions]):
        if i not in celltypes:
            celltypes.remove(i)
    cm = confusion_matrix(labels, predictions)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm = pd.DataFrame(cm, index=celltypes[:cm.shape[0]], columns=celltypes[:cm.shape[1]])
    nan_rows = [ind for i, ind in enumerate(cm.index) if all(cm.iloc[i].isna())]
    cm.drop(index=nan_rows, inplace=True)
    cm.to_csv(f"{save_dir}/confusion_matrix.csv")
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300)


def weighted_average(state_dicts, n_samples):
    sample_ratios = [n / sum(n_samples) for n in n_samples]
    global_weights = {}
    for param in state_dicts[0].keys():
        global_weights[param] = torch.stack(
            [state_dicts[i][param] * sample_ratios[i] for i in range(len(state_dicts))]).sum(0)
    return global_weights


def average_weights(state_dicts):
    global_weights = {}
    for param in state_dicts[0].keys():
        global_weights[param] = torch.stack(
            [state_dicts[i][param] for i in range(len(state_dicts))]).mean(0)
    return global_weights


def validate_fed_code(centralized_adata, clients_adata):
    for adata in clients_adata:
        print(adata.shape)
    # Concatenate client data
    adata_fed = anndata.concat(clients_adata, join='outer', label='batch')

    # Ensure the data matrices have the same shape for comparison
    if centralized_adata.shape != adata_fed.shape:
        raise ValueError(
            f"Centralized and federated data shapes do not match. Centralized shape: {centralized_adata.shape}, Federated shape: {adata_fed.shape}")

    # Calculate the absolute difference between the centralized and federated results
    diff = np.abs(centralized_adata.X - adata_fed.X)

    # Calculate the mean, sum, min, and max differences
    mean_diff = np.mean(diff)
    sum_diff = np.sum(diff)
    min_diff = np.min(diff)
    max_diff = np.max(diff)

    # Print the differences
    print("Mean difference between centralized and federated results:", mean_diff)
    print("Sum of differences between centralized and federated results:", sum_diff)
    print("Minimum difference between centralized and federated results:", min_diff)
    print("Maximum difference between centralized and federated results:", max_diff)


def split_data_by_batch(adata, batch_key, keep_vars):
    original_categories = {k: adata.obs[k].cat.categories for k in adata.obs.keys() if adata.obs[k].dtype == "category"}
    batch_ids = adata.obs[batch_key].tolist()
    unique_batch_ids = list(set(batch_ids))
    batches = {}
    for client, batch_id in enumerate(unique_batch_ids):
        batch_adata = adata[adata.obs[batch_key] == batch_id].copy()
        for k, v in original_categories.items():
            batch_adata.obs[k] = pd.Categorical(batch_adata.obs[k], categories=v)
        if keep_vars:
            batch_adata.var = adata.var.copy()
        batches[batch_id] = batch_adata
    return batches

def save_data_batches(batches: dict, data_dir: list or str, filename: str, keep_vars: bool = False):
    if type(data_dir) == str:
        data_dir = [f"{data_dir}/client_{i}"for i in batches.keys()]
    for client, batch_adata in enumerate(batches.values()):
        if not os.path.exists(data_dir[client]):
            print(f"{data_dir[client]} does not exist!")
            os.makedirs(data_dir[client], exist_ok=True)
        if "gene_name" in batch_adata.var.keys() and not keep_vars:
            batch_adata.var.drop(columns=["gene_name"], inplace=True)
        batch_adata.write_h5ad(f"{data_dir[client]}/{filename}")
    return data_dir


def compare_models(model1, model2):
    if isinstance(model1, dict) and isinstance(model2, dict):
        state_dict1, state_dict2 = model1, model2
    else:
        state_dict1, state_dict2 = model1.state_dict(), model2.state_dict()

    if len(state_dict1) != len(state_dict2):
        return False, "The models have different numbers of parameters."

    shape_discrepancies = []
    weight_discrepancies = []

    for (name1, param1), (name2, param2) in zip(state_dict1.items(), state_dict2.items()):
        if name1 != name2:
            return False, f"Layer names do not match: {name1} vs {name2}"
        if param1.shape != param2.shape:
            shape_discrepancies.append((name1, param1.shape, param2.shape))
        elif not torch.equal(param1, param2):
            weight_discrepancies.append(name1)

    if shape_discrepancies:
        return False, f"Shape discrepancies found: {shape_discrepancies}"

    if weight_discrepancies:
        if len(weight_discrepancies) > 20:  # Arbitrary threshold for too many discrepancies
            return False, f"Weight discrepancies found in {len(weight_discrepancies)} layers."
        else:
            return False, f"Weight discrepancies found in layers: {weight_discrepancies}"

    return True, "The models have the same weights and shapes."


def compare_batchnorm_stats(model1, model2):
    def get_modules_and_state_dict(model):
        if isinstance(model, dict):
            return [(name, None) for name in model.keys()], model
        else:
            return list(model.named_modules()), model.state_dict()

    modules1, state_dict1 = get_modules_and_state_dict(model1)
    modules2, state_dict2 = get_modules_and_state_dict(model2)

    for (name1, _), (name2, _) in zip(modules1, modules2):
        if name1 in state_dict1 and "running_mean" in name1 and "running_var" in name1:
            running_mean1 = state_dict1[name1.replace(".running_mean", "") + ".running_mean"]
            running_var1 = state_dict1[name1.replace(".running_var", "") + ".running_var"]
            running_mean2 = state_dict2[name2.replace(".running_mean", "") + ".running_mean"]
            running_var2 = state_dict2[name2.replace(".running_var", "") + ".running_var"]
            if not torch.equal(running_mean1, running_mean2) or not torch.equal(running_var1, running_var2):
                print(f"Discrepancy in BatchNorm layer {name1}")
                print(f"Model 1 - {name1} running_mean: {running_mean1}")
                print(f"Model 1 - {name1} running_var: {running_var1}")
                print(f"Model 2 - {name2} running_mean: {running_mean2}")
                print(f"Model 2 - {name2} running_var: {running_var2}")


def clients_have_same_wights(clients):
    return models_have_same_weights([c.model for c in clients])


def models_have_same_weights(models):
    first_model = models[0]
    for model in models[1:]:
        is_same, msg = compare_models(first_model, model)
        compare_batchnorm_stats(first_model, model)
        if not is_same:
            print(msg)
            return False, msg
    print("The models have the same weights and shapes.")
    return True, "The models have the same weights and shapes."


def compare_vocabulary(vocab1, vocab2):
    """
    Compare two GeneVocab vocabularies to ensure they are identical.

    Parameters:
    - vocab1, vocab2: GeneVocab objects representing vocabularies.

    Returns:
    - bool: True if vocabularies are identical, False otherwise.
    - str: Message describing any discrepancies.
    """
    def extract_vocab_dict(gene_vocab):
        # Extract the torchtext Vocab object
        vocab_obj = gene_vocab.vocab
        # Convert to a dictionary {word: index}
        vocab_dict = {word: vocab_obj[word] for word in vocab_obj.get_itos()}
        return vocab_dict

    vocab1_dict = extract_vocab_dict(vocab1)
    vocab2_dict = extract_vocab_dict(vocab2)

    if len(vocab1_dict) != len(vocab2_dict):
        return False, "Vocabularies have different lengths."

    for gene, index1 in vocab1_dict.items():
        index2 = vocab2_dict.get(gene)
        if index2 is None:
            return False, f"Gene '{gene}' is missing in the second vocabulary."
        if index1 != index2:
            return False, f"Different indices for gene '{gene}': {index1} vs {index2}"

    return True, "Vocabularies are consistent."


def verify_tokenization_consistency(client_tokenized_data_list):
    """
    Verifies the consistency of tokenization across different clients.

    Parameters:
    - client_tokenized_data_list: A list of dictionaries containing tokenized data from different clients.

    Returns:
    - bool: True if all tokenizations are consistent, False otherwise.
    - str: Message describing any discrepancies.
    """
    if not client_tokenized_data_list:
        return False, "No tokenized data provided."

    # Take the first client's tokenized data as the reference
    reference_data = client_tokenized_data_list[0]
    ref_feature_length = reference_data['genes'].shape[1]

    for idx, tokenized_data in enumerate(client_tokenized_data_list[1:], start=1):
        feature_length = tokenized_data['genes'].shape[1]

        if feature_length != ref_feature_length:
            return False, f"Inconsistent feature length for client {idx}: {feature_length} vs {ref_feature_length}"

        # Optional: Check if the tokenized content is the same
        if not np.array_equal(reference_data['genes'], tokenized_data['genes']):
            return False, f"Tokenized content differs for client {idx}"

    return True, "Tokenization is consistent across all clients."


class ResultsRecorder:
    def __init__(self, dataset, file_name='param_tuning', logger=None, verbose=False):
        self.results_file = file_name + '.csv'
        self.pickle_file = file_name + '.pkl'
        self.columns = ['Dataset', 'Round', 'Metric', 'Value', 'n_epochs']
        self.dataset = dataset
        self.results_df = self.load_or_create_dataframe()
        self.all_results = self.load_or_create_pickle()
        self.logger = logger if logger else print
        self.verbose = verbose

    def load_or_create_dataframe(self):
        """Load the DataFrame from a CSV file or create a new one if the file doesn't exist."""
        if os.path.exists(self.results_file):
            return pd.read_csv(self.results_file)
        else:
            return pd.DataFrame(columns=self.columns)

    def load_or_create_pickle(self):
        """Load the results dictionary from a pickle file or create a new one if the file doesn't exist."""
        if os.path.exists(self.pickle_file):
            with open(self.pickle_file, 'rb') as f:
                return pickle.load(f)
        else:
            return {}

    def update_dataframe(self, accuracy, precision, recall, macro_f1, round_number=None, n_epochs=None, dataset=None):
        """Update the DataFrame with new round results."""
        if dataset is None:
            dataset = self.dataset
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Macro_F1': macro_f1
        }
        self.logger(
            f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Macro F1: {macro_f1:.3f}")
        new_rows = pd.DataFrame([{
            'Round': round_number,
            'Metric': metric,
            'Value': value,
            'n_epochs': n_epochs,
            'Dataset': dataset
        } for metric, value in metrics.items()])
        self.results_df = pd.concat([self.results_df, new_rows], ignore_index=True)

    def save_dataframe(self):
        """Save the DataFrame to the CSV file."""
        self.results_df.to_csv(self.results_file, index=False)
        if self.verbose:
            self.logger(f"Data successfully saved to {self.results_file}")

    def update_pickle(self, predictions, labels, id_maps, epoch, round_number, dataset=None):
        """Update the pickle file with detailed results for each epoch and round."""
        if dataset is None:
            dataset = self.dataset

        if dataset not in self.all_results:
            self.all_results[dataset] = {}


        if 'id_maps' not in self.all_results[dataset]:
            self.all_results[dataset]['id_maps'] = id_maps
        else:
            assert self.all_results[dataset]['id_maps'] == id_maps, f"ID Maps mismatch for dataset {dataset}"

        # Initialize epoch if not present
        if epoch not in self.all_results[dataset]:
            self.all_results[dataset][epoch] = {}

        # Ensure round_number is not overwritten
        if round_number in self.all_results[dataset][epoch]:
            self.logger(f"Warning: Round {round_number} already exists for epoch {epoch} in dataset {dataset}.")

        self.all_results[dataset][epoch][round_number] = {'predictions': predictions, 'labels': labels}

    def save_pickle(self):
        """Save the detailed results dictionary to the pickle file."""
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(self.all_results, f)
        if self.verbose:
            self.logger(f"Detailed results successfully saved to {self.pickle_file}")

    def record_metrics(self, round_number, accuracy, precision, recall, macro_f1, n_epochs):
        """Record and save metrics in the DataFrame."""
        self.update_dataframe(accuracy, precision, recall, macro_f1, round_number, n_epochs)
        self.save_dataframe()

    def record_detailed_results(self, epoch, round_number, predictions, labels, id_maps):
        """Record and save detailed results in the pickle file."""
        self.update_pickle(predictions, labels, id_maps, epoch, round_number)
        self.save_pickle()

    def update(self, accuracy, precision, recall, macro_f1, predictions, labels, id_maps, round_number, n_epochs,
               dataset=None):
        if dataset is None:
            dataset = self.dataset
        self.update_dataframe(accuracy, precision, recall, macro_f1, round_number, n_epochs, dataset)
        self.update_pickle(predictions, labels, id_maps, n_epochs, round_number, dataset)
        self.save()

    def save(self):
        self.save_dataframe()
        self.save_pickle()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab, count_matrix, gene_ids, emb_style='<cls>', pad_value='<pad>', batch_ids=None):
        self.vocab = vocab
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.batch_ids = batch_ids
        self.emb_style = emb_style
        self.pad_value = pad_value

    def __len__(self):
        return len(self.count_matrix)

    def __getitem__(self, idx):
        row = self.count_matrix[idx]
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx]
        genes = self.gene_ids[nonzero_idx]
        # append <cls> token at the beginning
        genes = np.insert(genes, 0, self.vocab[self.emb_style])
        values = np.insert(values, 0, self.pad_value)
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).float()
        output = {
            "id": idx,
            "genes": genes,
            "expressions": values,
        }
        if self.batch_ids is not None:
            output["batch_labels"] = self.batch_ids[idx]
        return output


def plot_embedding(adata, query, cell_type_key, output_dir):
    def concat_reference_query(ref, query):
        n_ref_samples, n_query_samples = len(ref), len(query)
        adata_concat = query.concatenate(ref, batch_key="dataset")
        # mark the reference vs. query dataset
        adata_concat.obs["is_ref"] = ["Query"] * n_query_samples + ["Reference"] * n_ref_samples
        adata_concat.obs["is_ref"] = adata_concat.obs["is_ref"].astype("category")
        # mask the query dataset cell types
        adata_concat.obs[cell_type_key] = adata_concat.obs[cell_type_key].astype("category")
        adata_concat.obs[cell_type_key] = adata_concat.obs[cell_type_key].cat.add_categories(["To be predicted"])
        adata_concat.obs[cell_type_key][: n_query_samples] = "To be predicted"
        return adata_concat



    concat_adata = concat_reference_query(adata, query)

    # Compute neighbors and UMAP
    sc.pp.neighbors(concat_adata, use_rep="X_scGPT")
    sc.tl.umap(concat_adata)

    def custom_plot_umap(adata, color_by, file_name, legend='no_legend'):
        """
        Custom function to plot UMAP embeddings with or without legends.

        Args:
            adata (AnnData): The AnnData object containing the UMAP coordinates.
            color_by (list): List of column names in `adata.obs` to color the plots by.
            file_name (str): Name of the file to save the plot.
            legend (str): Either 'no_legend' or 'legend_only'.
        """
        # Create the UMAP plot
        fig, axes = plt.subplots(1, len(color_by), figsize=(len(color_by) * 5, 5))

        if len(color_by) == 1:
            axes = [axes]

        for ax, color in zip(axes, color_by):
            sc.pl.umap(adata, color=color, ax=ax, show=False, frameon=False)
            if legend == 'no_legend':
                ax.get_legend().remove()  # Remove the legend
            elif legend == 'legend_only':
                # Clear the data but keep the legend
                for coll in ax.collections:
                    coll.remove()
                ax.set_title('')
                ax.set_xlabel('')
                ax.set_ylabel('')
            else:
                raise ValueError(f"Invalid value for legend: {legend}")

        if legend == 'no_legend':
            plt.tight_layout()
        elif legend == 'legend_only':
            # Adjust the subplot parameters to include more space for the legend
            plt.subplots_adjust(right=0.5)
            for ax in axes:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])  # resize the plot

        plt.savefig(file_name, dpi=300)
        plt.close()

    sc.pp.neighbors(concat_adata, use_rep="X_scGPT")
    sc.tl.umap(concat_adata)

    # Plot UMAP with and without legends
    custom_plot_umap(concat_adata, color_by=["is_ref", cell_type_key], file_name=f"{output_dir}/embedding_umap_plot.png",
                     legend='no_legend')
    custom_plot_umap(concat_adata, color_by=["is_ref", cell_type_key], file_name=f"{output_dir}/embedding_umap_legend.png",
                     legend='legend_only')



def l2_sim(a, b):
    sims = -np.linalg.norm(a - b, axis=1)
    return sims

def get_similar_vectors(vector, ref, top_k=10):
    # sims = cos_sim(vector, ref)
    sims = l2_sim(vector, ref)

    top_k_idx = np.argsort(sims)[::-1][:top_k]
    return top_k_idx, sims[top_k_idx]


def eval_reference_mapping(gt, preds, output_dir="output", logger=None):
    if logger is None:
        logger = print
    # Calculate evaluation metrics
    res_dict = {
        "accuracy": accuracy_score(gt, preds),
        "precision": precision_score(gt, preds, average="macro"),
        "recall": recall_score(gt, preds, average="macro"),
        "macro_f1": f1_score(gt, preds, average="macro"),
    }

    # Print the evaluation metrics
    logger("Evaluation Metrics:")
    for key, value in res_dict.items():
        logger(f"{key.capitalize()}: {value:.4f}")

    # Save the evaluation metrics to a CSV file
    metrics_df = pd.DataFrame([res_dict])
    metrics_df.to_csv(f"{output_dir}/evaluation_metrics.csv", index=False)

    # Prepare confusion matrix
    y_true = gt
    y_pred = preds
    cell_type_list = np.unique(y_true)
    matrix = confusion_matrix(y_true, y_pred, labels=cell_type_list)
    matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]

    # Create a DataFrame for the confusion matrix
    df = pd.DataFrame(matrix, index=cell_type_list[:matrix.shape[0]], columns=cell_type_list[:matrix.shape[1]])

    # Create and save the clustermap
    ax = sns.clustermap(df,
                        cmap='Purples',
                        annot=True, fmt=".2f",
                        annot_kws={'size': 8},
                        vmin=0,
                        vmax=1,
                        row_cluster=False,
                        col_cluster=False,
                        figsize=(14, 14))

    clustermap_path = f"{output_dir}/confusion_matrix_clustermap.png"
    plt.savefig(clustermap_path)
    plt.close()


def compute_perturbation_metrics(
    results: Dict,
    ctrl_adata: AnnData,
    non_zero_genes: bool = False,
    return_raw: bool = False,
) -> Dict:
    """
    Method 'compute_perturbation_metrics' is adapted from:
    https://github.com/bowang-lab/scGPT/blob/7301b51a72f5db321fccebb51bc4dd1380d99023/scgpt/utils/util.py#L429
    Originally developed by Bo Wang Lab for scGPT. Please refer to the original source for more details.

    Given results from a model run and the ground truth, compute metrics

    Args:
        results (:obj:`Dict`): The results from a model run
        ctrl_adata (:obj:`AnnData`): The adata of the control condtion
        non_zero_genes (:obj:`bool`, optional): Whether to only consider non-zero
            genes in the ground truth when computing metrics
        return_raw (:obj:`bool`, optional): Whether to return the raw metrics or
            the mean of the metrics. Default is False.

    Returns:
        :obj:`Dict`: The metrics computed
    """
    from scipy.stats import pearsonr

    # metrics:
    #   Pearson correlation of expression on all genes, on DE genes,
    #   Pearson correlation of expression change on all genes, on DE genes,

    metrics_across_genes = {
        "pearson": [],
        "pearson_de": [],
        "pearson_delta": [],
        "pearson_de_delta": [],
    }

    metrics_across_conditions = {
        "pearson": [],
        "pearson_delta": [],
    }

    conditions = np.unique(results["pert_cat"])
    assert not "ctrl" in conditions, "ctrl should not be in test conditions"
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}

    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()  # (n_genes,)
    assert ctrl_adata.X.max() <= 1000, "gene expression should be log transformed"

    true_perturbed = results["truth"]  # (n_cells, n_genes)
    assert true_perturbed.max() <= 1000, "gene expression should be log transformed"
    true_mean_perturbed_by_condition = np.array(
        [true_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )  # (n_conditions, n_genes)
    true_mean_delta_by_condition = true_mean_perturbed_by_condition - mean_ctrl
    zero_rows = np.where(np.all(true_mean_perturbed_by_condition == 0, axis=1))[
        0
    ].tolist()
    zero_cols = np.where(np.all(true_mean_perturbed_by_condition == 0, axis=0))[
        0
    ].tolist()

    pred_perturbed = results["pred"]  # (n_cells, n_genes)
    pred_mean_perturbed_by_condition = np.array(
        [pred_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )  # (n_conditions, n_genes)
    pred_mean_delta_by_condition = pred_mean_perturbed_by_condition - mean_ctrl

    def corr_over_genes(x, y, conditions, res_list, skip_rows=[], non_zero_mask=None):
        """compute pearson correlation over genes for each condition"""
        for i, c in enumerate(conditions):
            if i in skip_rows:
                continue
            x_, y_ = x[i], y[i]
            if non_zero_mask is not None:
                x_ = x_[non_zero_mask[i]]
                y_ = y_[non_zero_mask[i]]
            res_list.append(pearsonr(x_, y_)[0])

    corr_over_genes(
        true_mean_perturbed_by_condition,
        pred_mean_perturbed_by_condition,
        conditions,
        metrics_across_genes["pearson"],
        zero_rows,
        non_zero_mask=true_mean_perturbed_by_condition != 0 if non_zero_genes else None,
    )
    corr_over_genes(
        true_mean_delta_by_condition,
        pred_mean_delta_by_condition,
        conditions,
        metrics_across_genes["pearson_delta"],
        zero_rows,
        non_zero_mask=true_mean_perturbed_by_condition != 0 if non_zero_genes else None,
    )

    def find_DE_genes(adata, condition, geneid2idx, non_zero_genes=False, top_n=20):
        """
        Find the DE genes for a condition
        """
        key_components = next(
            iter(adata.uns["rank_genes_groups_cov_all"].keys())
        ).split("_")
        assert len(key_components) == 3, "rank_genes_groups_cov_all key is not valid"

        condition_key = "_".join([key_components[0], condition, key_components[2]])

        de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
        if non_zero_genes:
            de_genes = adata.uns["top_non_dropout_de_20"][condition_key]
            # de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
            # de_genes = de_genes[adata.uns["non_zeros_gene_idx"][condition_key]]
            # assert len(de_genes) > top_n

        de_genes = de_genes[:top_n]

        de_idx = [geneid2idx[i] for i in de_genes]

        return de_idx, de_genes

    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))
    de_idx = {
        c: find_DE_genes(ctrl_adata, c, geneid2idx, non_zero_genes)[0]
        for c in conditions
    }
    mean_ctrl_de = np.array(
        [mean_ctrl[de_idx[c]] for c in conditions]
    )  # (n_conditions, n_diff_genes)

    true_mean_perturbed_by_condition_de = np.array(
        [
            true_mean_perturbed_by_condition[i, de_idx[c]]
            for i, c in enumerate(conditions)
        ]
    )  # (n_conditions, n_diff_genes)
    zero_rows_de = np.where(np.all(true_mean_perturbed_by_condition_de == 0, axis=1))[
        0
    ].tolist()
    true_mean_delta_by_condition_de = true_mean_perturbed_by_condition_de - mean_ctrl_de

    pred_mean_perturbed_by_condition_de = np.array(
        [
            pred_mean_perturbed_by_condition[i, de_idx[c]]
            for i, c in enumerate(conditions)
        ]
    )  # (n_conditions, n_diff_genes)
    pred_mean_delta_by_condition_de = pred_mean_perturbed_by_condition_de - mean_ctrl_de

    corr_over_genes(
        true_mean_perturbed_by_condition_de,
        pred_mean_perturbed_by_condition_de,
        conditions,
        metrics_across_genes["pearson_de"],
        zero_rows_de,
    )
    corr_over_genes(
        true_mean_delta_by_condition_de,
        pred_mean_delta_by_condition_de,
        conditions,
        metrics_across_genes["pearson_de_delta"],
        zero_rows_de,
    )

    if not return_raw:
        for k, v in metrics_across_genes.items():
            metrics_across_genes[k] = np.mean(v)
        for k, v in metrics_across_conditions.items():
            metrics_across_conditions[k] = np.mean(v)
    metrics = metrics_across_genes

    return metrics


def dump_perturbation_results(results: List, pool_size: int, save_file: str):
    """
    Dumps the perturbation results to a file, including additional control statistics.

    Parameters
    ----------
    results : List
        A list of tuples containing perturbation queries and their corresponding results (genes, truth, pred, control statistics).
    pool_size : int
        The size of the pool (number of control samples used).
    save_file : str
        The file path where the results will be saved.
    """
    data_to_save = {}
    for query, (genes, truth, pred, ctrl_mean) in results:
        data_to_save[query] = {
            "pool_size": pool_size,
            "genes": genes,
            "truth": truth,
            "pred": pred,
            "ctrl_mean": ctrl_mean
        }
    with open(save_file, "wb") as f:
        pickle.dump(data_to_save, f)


def dump_pert_subgroup_results(test_metrics, subgroup_analysis, save_file):
    # Step 8: Save everything to a single pickle file
    results_to_save = {
        "test_metrics": test_metrics,
        "subgroup_analysis": subgroup_analysis
    }

    with open(save_file, "wb") as f:
        pickle.dump(results_to_save, f)



def load_and_plot_perturbation_results(genes=None, truth=None, pred=None, ctrl_mean=None, plot_filename: str = None):
    """
    Load the results of a perturbation and create a plot comparing predicted and observed
    changes in gene expression over control.

    Parameters:
    ----------
    query : str
        The perturbation being analyzed (e.g., "gene1+ctrl").
    genes : list of str
        List of gene names.
    truth : np.ndarray
        The observed (ground truth) gene expression values for the perturbation condition.
    pred : np.ndarray
        The predicted gene expression values from the model for the same perturbation condition.
    pert_changes : np.ndarray
        The perturbation changes as calculated relative to control.
    ctrl_stats : dict
        A dictionary containing control statistics including mean, quartiles, and whiskers.
        - 'mean': Mean control expression for each gene.
    file_name : str
        File path to save the generated plot and data.
    """
    csv_filename = plot_filename.replace(".png", ".csv")
    # if os.path.exists(csv_filename):
    #     df = pd.read_csv(csv_filename)
    # else:
    df = create_pert_violin_plot_df(ctrl_mean, plot_filename, genes, pred, truth)
    plt.figure(figsize=(15, 5))
    plt.grid(False)
    sns.violinplot(x="Gene", y="Expression Change", hue="Type", data=df, inner="box",
                   inner_kws={'box_width': 2, 'whis_width': 3, 'marker': 'o'}, palette="muted", width=0.8)
    plt.axhline(0, linestyle="dashed", color="green")
    plt.ylabel("Expression Change", fontsize=20)
    plt.xlabel("")
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize=16)
    plt.subplots_adjust(bottom=0.3, top=0.9)
    plt.legend(loc='upper center', bbox_to_anchor=(0.22, 1.2), ncol=2, frameon=False, fontsize=20)
    plt.savefig(plot_filename, dpi=300)
    plt.show()
    plt.close()


def create_pert_violin_plot_df(ctrl_mean, csv_file, genes, pred, truth):
    pred_changes_to_ctrl_mean = pred - ctrl_mean
    truth_changes_to_ctrl_mean = truth - ctrl_mean
    # Create a DataFrame for the observed and predicted changes
    num_samples = truth_changes_to_ctrl_mean.shape[0]  # Number of samples
    # Create DataFrames with sample info
    truth_df = pd.DataFrame({
        "Gene": list(genes) * num_samples,
        "Expression Change": truth_changes_to_ctrl_mean.flatten(),
        "Sample": [f"Sample_{i}" for i in range(num_samples)] * len(genes),
        "Type": "Observed"
    })
    num_samples = pred_changes_to_ctrl_mean.shape[0]
    pred_df = pd.DataFrame({
        "Gene": list(genes) * num_samples,
        "Expression Change": pred_changes_to_ctrl_mean.flatten(),
        "Sample": [f"Sample_{i}" for i in range(num_samples)] * len(genes),
        "Type": "Predicted"
    })
    df = pd.concat([truth_df, pred_df])
    df.to_csv(csv_file, index=False)
    return df



def aggregate(fed_model, local_weights, n_local_samples, **kwargs):
    if fed_model.fed_config.aggregation_type == "FedAvg":
        fed_model.aggregate(local_weights)
    elif fed_model.fed_config.aggregation_type == "WeightedFedAvg":
        fed_model.weighted_aggregate(local_weights, n_local_samples)
    else:
        raise NotImplementedError(f"Aggregation type {fed_model.fed_config.aggregation_type} not implemented")


def concat_adata(adata1, adata2):
    assert all(adata1.var == adata2.var), "Control data variables do not match!"
    assert all(adata1.obs.keys() == adata2.obs.keys()), "Control data observations do not match!"
    assert adata1.uns.keys() == adata2.uns.keys(), "Control data uns keys do not match!"
    adata = anndata.concat([adata1, adata2])
    adata.uns = adata1.uns.copy()
    adata.var = adata2.var.copy()
    return adata


def plot_condition_heatmap(df, plt_name):
    plt.figure(figsize=(11, 10))
    value_to_int = {j: i for i, j in enumerate(['Unseen', 'Train', 'Valid', 'Test'])}
    n = len(value_to_int)
    cmap = sns.color_palette("light:slateblue", as_cmap=True)
    matrix = np.triu(df.values, 1)
    ax = sns.heatmap(df.replace(value_to_int), cmap=colors.ListedColormap(cmap(np.linspace(0, 1, 4))),
                     linewidths=0.05, mask=matrix)
    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=90)
    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
    colorbar.set_ticklabels(list(value_to_int.keys()))
    plt.savefig(plt_name, dpi=300)


class TopKMetricsEvaluator:
    """
    A class to evaluate top-k metrics such as precision, recall, top-k accuracy, and Average Reciprocal Rank (ARR) for gene predictions.

    Attributes
    ----------
    experiment_name : str
        The name of the experiment.
    experiment_parameters : Dict[str, Any]
        A dictionary of additional parameters related to the experiment.
    output_dir : str
        The directory where the output CSV files will be stored.
    append : bool, optional
        Whether to append to the output files if they already exist (default is True).
    predictions : List[List[int]]
        List of predicted gene sets for each cell (for batch processing).
    true_values : List[List[int]]
        List of actual perturbed gene sets for each cell (for batch processing).
    metrics : dict
        A dictionary to store top-k metrics like precision, recall, accuracy, hit rate for each k.
    arr_df : pd.DataFrame
        DataFrame to store the Average Reciprocal Rank (ARR).
    """

    def __init__(self, experiment_name: str,
                 experiment_parameters: Dict[str, Any],
                 output_dir: str,
                 predictions: List[List[int]] = None,
                 true_values: List[List[int]] = None,
                 append: bool = True):
        """
        Initialize the evaluator with predictions and true values (optional for gradual input).
        """
        self.predictions = predictions if predictions is not None else []
        self.true_values = true_values if true_values is not None else []
        self.experiment_name = experiment_name
        self.experiment_parameters = experiment_parameters
        self.output_dir = output_dir
        self.append = append
        self.topk_filepath = os.path.join(self.output_dir, f"{self.experiment_name}_top_k_metrics.csv")
        self.arr_filepath = os.path.join(self.output_dir, f"{self.experiment_name}_arr_metrics.csv")
        self.metrics = {'k': [], 'precision': [], 'recall': [], 'top_k_accuracy': [], 'hit_rate': [], 'min_hit': []}
        self.arr_df = pd.DataFrame()
        self.hit_range = []

    def add_data(self, predictions: List[int], true_values: List[int]):
        """
        Add predictions and true values incrementally for the gradual input scenario.

        Parameters
        ----------
        predictions : List[int]
            Predicted gene set for a single cell.
        true_values : List[int]
            True perturbed gene set for a single cell.
        """
        self.predictions.append(predictions)
        self.true_values.append(true_values)
        # Adjust hit range based on the new true_values
        self.hit_range = list(range(1, len(true_values) + 1))

    def arr_for_single_cell(self, pred: List[int], true: List[int]) -> float:
        """
        Calculate the Average Reciprocal Rank (ARR) for a single cell.
        """
        reciprocal_ranks = []
        for gene in true:
            if gene in pred:
                rank = pred.index(gene) + 1  # Get 1-based rank
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0

    def precision_at_k(self, k: int) -> float:
        """
        Calculate precision at k for all cells.
        """
        precisions = []
        for pred, true in zip(self.predictions, self.true_values):
            top_k_pred = set(pred[:k])
            relevant_in_k = top_k_pred.intersection(true)
            precisions.append(len(relevant_in_k) / k)
        return np.mean(precisions)

    def recall_at_k(self, k: int) -> float:
        """
        Calculate recall at k for all cells.
        """
        recalls = []
        for pred, true in zip(self.predictions, self.true_values):
            top_k_pred = set(pred[:k])
            relevant_in_k = top_k_pred.intersection(true)
            recalls.append(len(relevant_in_k) / len(true))
        return np.mean(recalls)

    def top_k_accuracy(self, k: int) -> float:
        """
        Calculate top-k accuracy for all cells.
        """
        accuracies = []
        for pred, true in zip(self.predictions, self.true_values):
            top_k_pred = set(pred[:k])
            accuracies.append(1 if len(top_k_pred.intersection(true)) > 0 else 0)
        return np.mean(accuracies)

    def hit_rate_at_k(self, k: int, min_hits: int = 1) -> float:
        """
        Calculate hit rate at k for all cells based on the number of hits.
        """
        hits = []
        for pred, true in zip(self.predictions, self.true_values):
            top_k_pred = set(pred[:k])
            num_hits = len(top_k_pred.intersection(true))
            hits.append(1 if num_hits >= min_hits else 0)
        return np.mean(hits)

    def update_metrics_for_k(self, k: int):
        """
        Gradually update metrics for a specific k value.
        """
        precision = self.precision_at_k(k)
        recall = self.recall_at_k(k)
        top_k_acc = self.top_k_accuracy(k)

        for min_hit in self.hit_range:
            self.metrics['k'].append(k)
            self.metrics['precision'].append(precision)
            self.metrics['recall'].append(recall)
            self.metrics['top_k_accuracy'].append(top_k_acc)
            self.metrics['min_hit'].append(min_hit)
            self.metrics['hit_rate'].append(self.hit_rate_at_k(k, min_hits=min_hit))

    def finalize_metrics(self):
        """
        Once all k values are provided, calculate ARR (k-independent) and store metrics into DataFrames.
        """
        # Convert the collected metrics into a DataFrame
        self.metrics_df = pd.DataFrame(self.metrics)

        # Calculate ARR and store it in a separate DataFrame
        self.arr_df = pd.DataFrame({'ARR': [self.average_reciprocal_rank()]})

        # Add additional columns for experiment parameters
        self.add_exp_specific_columns()

    def average_reciprocal_rank(self) -> float:
        """
        Calculate the Average Reciprocal Rank (ARR) across all cells.

        Returns
        -------
        float
            The ARR across all cells.
        """
        arrs = [self.arr_for_single_cell(pred, true) for pred, true in zip(self.predictions, self.true_values)]
        return np.mean(arrs) if arrs else 0

    def calculate_metrics_for_all_ks(self, k_range: List[int]):
        """
        Calculate metrics for multiple k values and store them in a DataFrame.

        Parameters
        ----------
        k_range : list
            List of k values to calculate metrics for (e.g., range(1, 16)).
        """
        metrics = {'k': [], 'precision': [], 'recall': [], 'top_k_accuracy': [], 'hit_rate': [], 'min_hit': []}
        for k in k_range:
            precision = self.precision_at_k(k)
            recall = self.recall_at_k(k)
            top_k_acc = self.top_k_accuracy(k)
            for min_hit in self.hit_range:
                metrics['k'].append(k)
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['top_k_accuracy'].append(top_k_acc)
                metrics['min_hit'].append(min_hit)
                metrics['hit_rate'].append(self.hit_rate_at_k(k, min_hits=min_hit))
        self.metrics_df = pd.DataFrame(metrics)
        self.arr_df = pd.DataFrame({'ARR': [self.average_reciprocal_rank()]})
        self.add_exp_specific_columns()

    def add_exp_specific_columns(self):
        """
        Add extra columns (experiment parameters and name) to the DataFrames.
        """
        if self.metrics_df.empty or self.arr_df.empty:
            raise ValueError("Metrics dataframe is empty.")

        for key, value in self.experiment_parameters.items():
            self.metrics_df[key] = value
            self.arr_df[key] = value

        self.metrics_df['experiment'] = self.experiment_name
        self.arr_df['experiment'] = self.experiment_name

    def write_metrics_to_csv(self):
        """
        Write the metrics DataFrames to CSV files.
        """
        if not self.metrics_df.empty:
            mode = 'a' if self.append else 'w'
            write_header = not os.path.exists(self.topk_filepath) or not self.append
            self.metrics_df.to_csv(self.topk_filepath, mode=mode, header=write_header, index=False)
            self.arr_df.to_csv(self.arr_filepath, mode=mode, header=write_header, index=False)

    def read_metrics_from_csv(self):
        """
        Read the metrics from CSV files into DataFrames.
        """
        self.metrics_df = pd.read_csv(self.topk_filepath)
        self.arr_df = pd.read_csv(self.arr_filepath)


class GPUUsageTracker:
    def __init__(self, device_index: int = None):
        if device_index is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = get_cuda_device(device_index)

        # Initial memory stats
        self.start_allocated = torch.cuda.memory_allocated(self.device)
        self.start_reserved = torch.cuda.memory_reserved(self.device)
        self.start_max_allocated = torch.cuda.max_memory_allocated(self.device)
        self.start_max_reserved = torch.cuda.max_memory_reserved(self.device)

        print(f"GPU Tracking Started on {self.device}.")
        self.print_memory_stats("At start")

    def print_memory_stats(self, context=""):
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)

        print(f"{context} - Allocated Memory: {allocated:.2f} MB")
        print(f"{context} - Reserved Memory: {reserved:.2f} MB")
        print(f"{context} - Max Allocated Memory: {max_allocated:.2f} MB")
        print(f"{context} - Max Reserved Memory: {max_reserved:.2f} MB")

    def generate_report(self):
        print("\nGenerating GPU usage report:")
        self.print_memory_stats("At end")

        print("\nGPU Usage Summary:")
        print(
            f"Memory allocated increased by: {(torch.cuda.memory_allocated(self.device) - self.start_allocated) / (1024 ** 2):.2f} MB")
        print(
            f"Memory reserved increased by: {(torch.cuda.memory_reserved(self.device) - self.start_reserved) / (1024 ** 2):.2f} MB")
        print(
            f"Max memory allocated increased by: {(torch.cuda.max_memory_allocated(self.device) - self.start_max_allocated) / (1024 ** 2):.2f} MB")
        print(
            f"Max memory reserved increased by: {(torch.cuda.max_memory_reserved(self.device) - self.start_max_reserved) / (1024 ** 2):.2f} MB")
