import os

import anndata
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import scgpt as scg
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Dict, Tuple, List
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
import dataclasses
import yaml
import scanpy as sc
import pickle
import matplotlib.pyplot as plt
import logging
import sys
import random

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)

set_seed()

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


@dataclasses.dataclass
class LogConfig:
    log_interval: int
    save_eval_interval: int
    do_eval_scib_metrics: bool
    retain_best_model: bool


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
    # TODO: check this comment does not affect ms dataset
    # adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    # adata.var.set_index(adata.var["gene_name"], inplace=True)
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

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
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
    palette_ = palette_ * 3  # Extend the palette if needed
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


def split_data_by_batch(adata, batch_key):
    original_categories = {k: adata.obs[k].cat.categories for k in adata.obs.keys() if adata.obs[k].dtype == "category"}
    batch_ids = adata.obs[batch_key].tolist()
    unique_batch_ids = list(set(batch_ids))
    batches = {}
    for client, batch_id in enumerate(unique_batch_ids):
        batch_adata = adata[adata.obs[batch_key] == batch_id].copy()
        for k, v in original_categories.items():
            batch_adata.obs[k] = pd.Categorical(batch_adata.obs[k], categories=v)
        batches[batch_id] = batch_adata
    return batches

def save_data_batches(batches: dict, data_dir: list or str, filename: str):
    if type(data_dir) == str:
        data_dir = [f"{data_dir}/client_{i}"for i in batches.keys()]
    for client, batch_adata in enumerate(batches.values()):
        if not os.path.exists(data_dir[client]):
            print(f"{data_dir[client]} does not exist!")
            os.makedirs(data_dir[client], exist_ok=True)
        if "gene_name" in batch_adata.var.keys():
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