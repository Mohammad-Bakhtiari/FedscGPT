from torch.utils.data import DataLoader
import torch
from typing import Dict, List
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shutil
import json
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from FedscGPT.utils import SeqDataset, dump_results, plot, ResultsRecorder
from FedscGPT.centralized.models import ScGPT
from FedscGPT.utils import read_h5ad
import copy
from functools import partial

class Base(ScGPT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.dataset.data_is_raw = False
        self.config.dataset.filter_gene_by_counts = False
        self.preprocessor = None

    def get_raw_testset(self, query_adata):
        if query_adata is None:
            return None
        return read_h5ad(self.data_dir, query_adata)

    def harmonize(self, adata, adata_test=None):
        self.manege_id2type(adata)
        self.set_obs_and_vars(adata)
        if adata_test is not None:
            self.set_obs_and_vars(adata_test)
            self.check_category(adata, adata_test, self.celltype_key)
        else:
            self.log("No query dataset is provided.")

    def manege_id2type(self, adata):
        self.unique_cell_types = adata.obs[self.celltype_key].cat.categories.tolist()
        self.config.model.n_cls = len(self.unique_cell_types)
        self.cell_id2type = dict(enumerate(self.unique_cell_types))

    def set_obs_and_vars(self, adata):
        adata.obs["batch_id"] = adata.obs[self.batch_key].astype("category").cat.codes.values
        adata.obs["celltype_id"] = adata.obs[self.celltype_key].astype("category").cat.codes.values
        adata.var["gene_name"] = adata.var.index.tolist()

    def harmonize_query(self, adata_test):
        self.manege_id2type(adata_test)
        self.set_obs_and_vars(adata_test)

    def check_category(self, reference, query, obs_key):
        if reference.obs[obs_key].dtype != "category":
            self.log(f"{obs_key} is not a category in the reference dataset.")
        if query.obs[obs_key].dtype != "category":
            self.log(f"{obs_key} is not a category in the query dataset.")
        if all(reference.obs[obs_key].cat.categories != query.obs[obs_key].cat.categories):
            self.log(f"Categories of {obs_key} are not the same in the reference and query datasets.")
            reference_unique_obs = reference.obs[obs_key].unique().tolist()
            query_unique_cell_obs = query.obs[obs_key].unique().tolist()
            self.log(f"Unique {obs_key} in the reference dataset: {reference_unique_obs}")
            reference.obs[obs_key] = reference.obs[obs_key].cat.set_categories(query_unique_cell_obs)
            query.obs[obs_key] = query.obs[obs_key].cat.set_categories(reference_unique_obs)

    def load_pretrained_config(self):
        model_config_file = os.path.join(self.pretrained_model_dir, 'args.json')
        model_file = os.path.join(self.pretrained_model_dir, 'best_model.pt')
        vocab_file = os.path.join(self.pretrained_model_dir, 'vocab.json')
        self.vocab = GeneVocab.from_file(vocab_file)
        if self.pretrained_model_dir != self.output_dir:
            output_dir = os.path.join(self.output_dir, 'model')
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            shutil.copy(vocab_file, os.path.join(output_dir, 'vocab.json'))
            shutil.copy(model_config_file, os.path.join(output_dir, "args.json"))
        self.add_special_tokens()
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        self.log(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        self.config.pretrained_model = {**model_configs}
        self.config.model.embsize = model_configs["embsize"]
        self.config.model.nhead = model_configs["nheads"]
        self.config.model.d_hid = model_configs["d_hid"]
        self.config.model.nlayers = model_configs["nlayers"]

    def filter_id_in_vocab(self, adata):
        adata.var["id_in_vocab"] = [1 if gene in self.vocab else -1 for gene in adata.var["gene_name"]]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        self.log(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(self.vocab)}."
        )
        filtered_adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()
        for column in adata.obs.select_dtypes(['category']).columns:
            original_categories = adata.obs[column].cat.categories
            filtered_adata.obs[column] = filtered_adata.obs[column].cat.set_categories(original_categories)
        return filtered_adata

    def filter(self, adata, adata_test=None):
        self.log("Filtering genes in the reference dataset that are not in the vocabulary.")
        adata = self.filter_id_in_vocab(adata)
        if adata_test is not None:
            self.log("Filtering genes in the query dataset that are not in the vocabulary.")
            adata_test = self.filter_id_in_vocab(adata_test)
            return adata, adata_test
        return adata

    def add_special_tokens(self):
        for s in self.special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)

    def instantiate_preprocessor(self):
        self.preprocessor = Preprocessor(
            use_key=self.config.dataset.raw_data_key,
            filter_gene_by_counts=self.config.dataset.filter_gene_by_counts,
            filter_cell_by_counts=self.config.dataset.filter_cell_by_counts,
            normalize_total=self.config.dataset.normalize_total,
            result_normed_key=self.config.dataset.result_normed_key,
            log1p=self.config.dataset.log1p,
            result_log1p_key=self.config.dataset.result_log1p_key,
            subset_hvg=self.config.dataset.subset_hvg,
            hvg_flavor=self.config.dataset.hvg_flavor,
            binning=self.config.preprocess.n_bins,
            result_binned_key=self.config.dataset.result_binned_key,
        )

    def set_layer_key(self):
        self.input_layer_key = {
            "normed_raw": "X_normed",
            "log1p": "X_normed",
            "binned": "X_binned",
        }[self.config.preprocess.input_style]

    def get_all_counts(self, adata):
        try:
            return adata.layers[self.input_layer_key].A if issparse(adata.layers[self.input_layer_key]) else adata.layers[
                self.input_layer_key]
        except:

            if len(adata.layers.keys()) == 0:
                msg = "adata.layers is empty. Make sure the data is preprocessed!"
            else:
                msg = f"{self.input_layer_key} is not in adata.layers.keys()!"
            raise ValueError(msg)

class Training(Base):
    def __init__(self, reference_adata, **kwargs):
        super().__init__(**kwargs)
        self.read_reference(reference_adata)

    def read_reference(self, reference_adata):
        self.adata = read_h5ad(self.data_dir, reference_adata)

    def preprocess_reference(self):
        self.preprocessor(self.adata, batch_key=None)

    def post_prep(self):
        self.set_layer_key()
        all_counts = self.get_all_counts(self.adata)
        celltypes_labels = self.adata.obs["celltype_id"].tolist()
        celltypes_labels = np.array(celltypes_labels)
        batch_ids = self.adata.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)
        (
            self.train_data,
            self.valid_data,
            self.train_celltype_labels,
            self.valid_celltype_labels,
            self.train_batch_labels,
            self.valid_batch_labels,
        ) = train_test_split(
            all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
        )


class Inference(Base):
    def __init__(self, query_adata, dataset_name, param_tuning_res, load_model=True, model_name="model.pt", param_tuning=False, **kwargs):
        super().__init__(**kwargs)
        self.celltypes_labels = None
        self.read_query(query_adata)
        self.manege_id2type(self.adata_test)
        self.set_obs_and_vars(self.adata_test)
        self.load_pretrained_config()
        self.adata_test = self.filter_id_in_vocab(self.adata_test)
        self.instantiate_preprocessor()
        self.preprocess_query()
        self.gene_ids = np.array(self.vocab(self.adata_test.var["gene_name"].tolist()), dtype=int)
        self.instantiate_transformer_model()
        if load_model:
            self.load_pretrained_model(model_name)
        else:
            self.model.to(self.device)
        self.setup_losses()
        self.best_model = copy.deepcopy(self.model)
        self.best_model.eval()
        self.plot_dir = f"{self.output_dir}/plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir, exist_ok=True)
        self.test_loader = None
        self.param_tuning = param_tuning
        self.result_recorder = ResultsRecorder(dataset=dataset_name, file_name=param_tuning_res, logger=self.log)

    def read_query(self, query_adata):
        self.adata_test_raw = read_h5ad(self.data_dir, query_adata)
        self.adata_test = self.adata_test_raw.copy()

    def preprocess_query(self):
        self.preprocessor(self.adata_test, batch_key=None)
        self.set_layer_key()

    def test(self, round_num, n_epochs) -> (np.ndarray, np.ndarray, Dict[str, float]):
        if self.test_loader is None or self.celltypes_labels is None:
            self.load_test_loader()
        predictions = self.evaluate(self.best_model, loader=self.test_loader, return_raw=True)
        accuracy = accuracy_score(self.celltypes_labels, predictions)
        precision = precision_score(self.celltypes_labels, predictions, average="macro")
        recall = recall_score(self.celltypes_labels, predictions, average="macro")
        macro_f1 = f1_score(self.celltypes_labels, predictions, average="macro")
        self.update_records(accuracy=accuracy, precision=precision, recall=recall, macro_f1=macro_f1, round_number=round_num, n_epochs=n_epochs, predictions=predictions)
        results = {
            "test/accuracy": accuracy,
            "test/precision": precision,
            "test/recall": recall,
            "test/macro_f1": macro_f1,
        }
        return predictions, self.celltypes_labels, results

    def load_test_loader(self):
        all_counts = self.get_all_counts(self.adata_test)
        self.celltypes_labels = np.array(self.adata_test.obs["celltype_id"].tolist())
        batch_ids = self.adata_test.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)
        tokenized_test = self.tokenize_and_pad_batch(all_counts)
        input_values_test = self.random_mask_value(tokenized_test["values"])
        test_data_pt = {
            "gene_ids": tokenized_test["genes"],
            "values": input_values_test,
            "target_values": tokenized_test["values"],
            "batch_labels": torch.from_numpy(batch_ids).long(),
            "celltype_labels": torch.from_numpy(self.celltypes_labels).long(),
        }
        self.test_loader = DataLoader(
            dataset=SeqDataset(test_data_pt),
            batch_size=self.config.train.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), self.config.train.eval_batch_size // 2),
            pin_memory=True,
        )

    def inference(self, plot_results=True, save=True, round_num=None, n_epochs=None):
        predictions, labels, results = self.test(round_num, n_epochs)
        self.adata_test_raw.obs["predictions"] = [self.cell_id2type[p] for p in predictions]
        if plot_results:
            plot(self.adata_test_raw, self.unique_cell_types, self.celltype_key, self.plot_dir)
        if save:
            self.save_results(labels, predictions, results)
        return predictions, labels

    def save_results(self, labels, predictions, results):
        dump_results(predictions, labels, results, self.cell_id2type, self.output_dir)

    def save_records(self):
        if self.param_tuning:
            self.result_recorder.save()

    def update_records(self, **kwargs):
        self.result_recorder.update(labels=self.celltypes_labels, id_maps=self.cell_id2type, **kwargs)

class CellTypeAnnotator(Training, Inference):
    def __init__(self, reference_adata, query_adata=None, **kwargs):
        Base.__init__(self, **kwargs)
        self.read_reference(reference_adata)
        if query_adata is not None:
            self.read_query(query_adata)
