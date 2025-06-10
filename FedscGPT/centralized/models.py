import json
import warnings

from torch.utils.data import DataLoader
import time
import torch
from typing import Dict, List
from scgpt import SubsetsBatchSampler
import os
import copy
import numpy as np
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value

from scgpt.model import TransformerModel
import shutil
from FedscGPT.base import BaseMixin
from FedscGPT.utils import SeqDataset, seed_worker, read_h5ad


class ScGPT(BaseMixin):
    """
    The main class for training and evaluating the model.
    cell_id2type: Dict[int, str] - a dictionary mapping cell type id to cell type name. Useful only for evaluation
    unique_cell_types: np.ndarray - a list of unique cell types
    special_tokens = ["<pad>", "<cls>", "<eoc>"] - special tokens used in the model
    """
    cell_id2type: Dict[int, str]
    unique_cell_types: List[str]
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    vocab: GeneVocab
    tokenized_train: Dict[str, torch.Tensor]
    tokenized_valid: Dict[str, torch.Tensor]
    gene_ids: np.ndarray
    input_layer_key: str
    best_model: TransformerModel
    best_model_epoch: int

    def __init__(self, data_dir, pretrained_model_dir, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.pretrained_model_dir = pretrained_model_dir

        self.sanity_check()
        self.check_input_style()
        self.train_kwarg = {
                            "CLS": self.config.train.CLS,
                            "CCE": self.config.train.CCE,
                            "MVC": self.config.train.MVC,
                            "ECS": self.config.train.ECS,
                            "do_sample": self.config.train.do_sample_in_train,
        }

    def read_reference(self, reference_adata):
        self.adata = read_h5ad(self.data_dir, reference_adata)

    def check_input_style(self):
        if self.config.preprocess.input_style == "category":
            self.config.preprocess.mask_value = self.config.preprocess.n_bins + 1
            self.config.preprocess.pad_value = self.config.preprocess.n_bins
            self.config.model.n_input_bins = self.config.preprocess.n_bins + 2
        else:
            self.config.preprocess.mask_value = -1
            self.config.preprocess.pad_value = -2
            self.config.model.n_input_bins = self.config.preprocess.n_bins

    def create_vocabulary(self):
        self.vocab = Vocab(
            VocabPybind(self.adata.var["gene_name"].tolist() + self.config.preprocess.special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
        self.vocab.set_default_index(self.vocab["<pad>"])

    def tokenize_and_pad_batch(self, data):
        return tokenize_and_pad_batch(
            data,
            self.gene_ids,
            max_len=self.config.preprocess.max_seq_len,
            vocab=self.vocab,
            pad_token=self.config.preprocess.pad_token,
            pad_value=self.config.preprocess.pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=self.config.preprocess.include_zero_gene,
        )

    def load_pretrained_config(self, set_pretrained_config=True):
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
        if set_pretrained_config:
            self.config.pretrained_model = {**model_configs}
            self.config.model.embsize = model_configs["embsize"]
            self.config.model.nhead = model_configs["nheads"]
            self.config.model.d_hid = model_configs["d_hid"]
            self.config.model.nlayers = model_configs["nlayers"]
            self.config.model.nlayers_cls = model_configs["n_layers_cls"]
            self.config.model.dropout = model_configs["dropout"]
            self.config.preprocess.pad_token = model_configs["pad_token"]
            self.config.preprocess.pad_value = model_configs["pad_value"]

    def add_special_tokens(self):
        for s in self.special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)

    def tokenize(self):
        self.gene_ids = np.array(self.vocab(self.adata.var["gene_name"].tolist()), dtype=int)
        self.tokenized_train = self.tokenize_and_pad_batch(self.train_data)
        self.log(
            f"train set number of samples: {self.tokenized_train['genes'].shape[0]}, "
            f"\n\t feature length: {self.tokenized_train['genes'].shape[1]}"
        )
        if self.valid_data:
            self.tokenized_valid = self.tokenize_and_pad_batch(self.valid_data)

            self.log(
                f"valid set number of samples: {self.tokenized_valid['genes'].shape[0]}, "
                f"\n\t feature length: {self.tokenized_valid['genes'].shape[1]}"
            )

    def instantiate_transformer_model(self):
        kwargs = copy.deepcopy(self.config.model.__dict__)
        self.model = TransformerModel(
            len(self.vocab),
            d_model=kwargs.pop("embsize"),
            vocab=self.vocab,
            pad_token=self.config.preprocess.pad_token,
            pad_value=self.config.preprocess.pad_value,
            **kwargs,
        )

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

    def load_pretrained_model(self, model_name="best_model.pt"):
        save_init_weights = True
        if self.init_weights_dir:
            save_init_weights = self.load_init_weights()
        if save_init_weights and self.pretrained_model_dir:
            model_dir = os.path.join(self.pretrained_model_dir, model_name)
            try:
                self.model.load_state_dict(torch.load(model_dir))
                self.log(f"Loading all model params from {self.pretrained_model_dir}")
            except:
                # only load params that are in the model and match the size
                self.load_matched_param(model_dir)
            self.save_init_weights()
        self.freeze_params()
        self.model.to(self.device)


    def train_for_epoch(self, loader, epoch) -> None:
        """
        Train the model for one epoch.
        """
        self.model.train()
        self.loss_meter.reset()
        num_batches = len(loader)
        if self.config.log.log_interval > num_batches:
            self.config.log.log_interval = num_batches
            self.log(f"Setting log_interval to {num_batches}")
        for batch, batch_data in enumerate(loader, 1):
            self.train_on_batch(batch_data)
            if batch % self.config.log.log_interval == 0 and batch > 0:
                lr = self.lr_schedulers['main'].get_last_lr()[0]
                log_txt = self.loss_meter.log(self.config.log.log_interval)
                log_txt = f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | lr {lr:05.4f} | " + log_txt
                self.log(log_txt)
                self.loss_meter.reset()

    def train_on_batch(self, batch_data):
        batch_labels, celltype_labels, input_gene_ids, input_values, target_values = self.unwrap_batch_data(batch_data)
        src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.preprocess.pad_token])
        with torch.cuda.amp.autocast(enabled=self.config.train.amp):
            output_dict = self.model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if self.config.train.INPUT_BATCH_LABELS or self.config.train.DSBN else None,
                **self.train_kwarg
            )

            masked_positions = input_values.eq(self.config.preprocess.mask_value)  # the postions to predict
            args_dict = {"batch_labels": batch_labels,
                         "celltype_labels": celltype_labels,
                         "masked_positions": masked_positions,
                         "target_values": target_values,
                         **output_dict}
            self.apply_loss(**args_dict)
            self.fedprox()

        self.model.zero_grad()
        self.scaler.scale(self.loss_meter.batch_loss).backward()
        self.scaler.unscale_(self.optimizers["main"])
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0,
                error_if_nonfinite=False if self.scaler.is_enabled() else True,
            )
            if len(w) > 0:
                self.logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {self.scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        self.scaler.step(self.optimizers["main"])
        self.scaler.update()
        if self.config.train.ADV:
            self.adversarial_training(batch_labels, input_gene_ids, input_values, src_key_padding_mask)
        self.loss_meter.reset_batch_loss()

    def fedprox(self):
        if self.use_fedprox and self.global_model:
            prox_term = 0
            for param, global_param in zip(self.model.parameters(), self.global_model.values()):
                prox_term += ((param - global_param.to(self.device)) ** 2).sum()
            self.loss_meter.batch_loss += (self.mu / 2) * prox_term

    def unwrap_batch_data(self, batch_data):
        input_gene_ids = batch_data["gene_ids"].to(self.device)
        input_values = batch_data["values"].to(self.device)
        target_values = batch_data["target_values"].to(self.device)
        batch_labels = batch_data["batch_labels"].to(self.device)
        celltype_labels = batch_data["celltype_labels"].to(self.device)
        return batch_labels, celltype_labels, input_gene_ids, input_values, target_values

    def adversarial_training(self, batch_labels, input_gene_ids, input_values, src_key_padding_mask, epoch):
        # rerun the model for adversarial training
        output_dict = self.model(
            input_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=batch_labels if self.config.train.INPUT_BATCH_LABELS or self.config.train.DSBN else None,
            CLS=self.config.train.CLS,
            CCE=self.config.train.CCE,
            MVC=self.config.train.MVC,
            ECS=self.config.train.ECS,
            do_sample=self.config.train.do_sample_in_train,
            # generative_training=False
        )
        # TRAINING DISCRIMINATOR
        loss_adv_D = self.losses['adv'](
            self.discriminator(output_dict["cell_emb"].detach()), batch_labels
        )
        if epoch > self.config.train.ADV.D_delay_epochs:
            self.discriminator.zero_grad()
            loss_adv_D.backward()
            self.optimizers['D'].step()
        # TRAINING ENCODER
        loss_adv_E = -self.losses['adv'](
            self.discriminator(output_dict["cell_emb"]), batch_labels
        )
        # NOTE: the loss is negative here because we want to maximize
        # the cross_entropy_loss, in other words, disguise against the discriminator
        if epoch > self.config.train.ADV.E_delay_epochs:
            self.model.zero_grad()
            self.discriminator.zero_grad()
            loss_adv_E.backward()
            self.optimizers['E'].step()
        self.loss_meter.update("adv_D", loss_adv_D.item())
        self.loss_meter.update("adv_E", loss_adv_E.item())

    def evaluate(self, model, loader: DataLoader, return_raw: bool = False) -> float:
        """
        Evaluate the model on the evaluation data.
        """
        model.eval()
        total_loss, total_dab, total_num, total_error = 0.0, 0.0, 0, 0.0
        predictions = []
        with (torch.no_grad()):
            for batch_data in loader:
                batch_labels, celltype_labels, input_gene_ids, input_values, target_values = \
                    self.unwrap_batch_data(batch_data)
                src_key_padding_mask = input_gene_ids.eq(self.vocab[self.config.preprocess.pad_token])
                with torch.cuda.amp.autocast(enabled=self.config.train.amp):
                    output_dict = model(input_gene_ids,
                                        input_values,
                                        src_key_padding_mask=src_key_padding_mask,
                                        batch_labels=batch_labels if self.config.train.INPUT_BATCH_LABELS or
                                                                     self.config.train.DSBN else None,
                                        CLS=self.config.train.CLS,  # evaluation does not need CLS or CCE
                                        CCE=False,
                                        MVC=False,
                                        ECS=False,
                                        do_sample=self.config.train.do_sample_in_train)
                    loss = self.losses['cls'](output_dict["cls_output"], celltype_labels)
                    total_loss += loss.item() * len(input_gene_ids)
                    if self.config.train.DAB:
                        loss_dab = self.losses['dab'](output_dict["dab_output"], batch_labels)
                        total_dab += loss_dab.item() * len(input_gene_ids)

                accuracy = (output_dict["cls_output"].argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_num += len(input_gene_ids)
                preds = output_dict["cls_output"].argmax(1).cpu().numpy()
                predictions.append(preds)
        if return_raw:
            return np.concatenate(predictions, axis=0)
        return total_loss / total_num, total_error / total_num

    def random_mask_value(self, tokenized_values):
        return random_mask_value(
            tokenized_values,
            mask_ratio=self.config.preprocess.mask_ratio,
            mask_value=self.config.preprocess.mask_value,
            pad_value=self.config.preprocess.pad_value,
        )

    def _prepare_split(self, tokenized_split, batch_labels, celltype_labels, epoch, train_split=True):
        masked_values = self.random_mask_value(tokenized_split["values"])
        if train_split:
            self.log(
                f"random masking at epoch {epoch:3d}, ratio of masked values in train:"
                f" {(masked_values == self.config.preprocess.mask_value).sum() / (masked_values - self.config.preprocess.pad_value).count_nonzero():.4f}",
            )

        tensor_batch_labels = torch.from_numpy(batch_labels).long()
        tensor_celltype_labels = torch.from_numpy(celltype_labels).long()
        input_gene_ids = tokenized_split["genes"]
        target_values = tokenized_split["values"]
        input_values = masked_values
        if self.config.preprocess.per_seq_batch_sample:  # TODO: update to random pick seq source in each traning batch
            sort_ids = np.argsort(batch_labels)
            input_gene_ids = input_gene_ids[sort_ids]
            input_values = input_values[sort_ids]
            target_values = target_values[sort_ids]
            tensor_batch_labels = tensor_batch_labels[sort_ids]
            tensor_celltype_labels = tensor_celltype_labels[sort_ids]


        return {
            "gene_ids": input_gene_ids,
            "values": input_values,
            "target_values": target_values,
            "batch_labels": tensor_batch_labels,
            "celltype_labels": tensor_celltype_labels,
        }

    def per_epoch_data_prep(self, epoch):
        train_data_pt = self._prepare_split(self.tokenized_train,
                                            self.train_batch_labels,
                                            self.train_celltype_labels,
                                            epoch)
        if self.tokenized_valid:
            valid_data_pt = self._prepare_split(self.tokenized_valid,
                                                self.valid_batch_labels,
                                                self.valid_celltype_labels,
                                                epoch,
                                                train_split=False)
        else:
            valid_data_pt = None
        return train_data_pt, valid_data_pt

    def per_epoch_dataloader(self,
                             data_pt: Dict[str, torch.Tensor],
                             batch_size: int,
                             shuffle: bool = False,
                             intra_domain_shuffle: bool = False,
                             drop_last: bool = False,
                             num_workers: int = 0,
                             ) -> DataLoader:
        if num_workers == 0:
            num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

        dataset = SeqDataset(data_pt)

        if self.config.preprocess.per_seq_batch_sample:
            # find the indices of samples in each seq batch
            subsets = []
            batch_labels_array = data_pt["batch_labels"].numpy()
            for batch_label in np.unique(batch_labels_array):
                batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
                subsets.append(batch_indices)
            data_loader = DataLoader(
                dataset=dataset,
                batch_sampler=SubsetsBatchSampler(
                    subsets,
                    batch_size,
                    intra_subset_shuffle=intra_domain_shuffle,
                    inter_subset_shuffle=shuffle,
                    drop_last=True,
                ),
                drop_last=True,
                num_workers=num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
            )
            return data_loader

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
        return data_loader

    def train(self):
        best_val_loss = float("inf")
        for epoch in range(1, self.config.train.epochs + 1):
            epoch_start_time = time.time()
            train_data_pt, valid_data_pt = self.per_epoch_data_prep(epoch)
            train_loader = self.per_epoch_dataloader(train_data_pt,
                                                     batch_size=self.config.train.batch_size,
                                                     shuffle=False,
                                                     intra_domain_shuffle=True,
                                                     drop_last=False)


            self.train_for_epoch(train_loader, epoch)
            if self.config.retain.best_model:
                num_eval_data = len(valid_data_pt["gene_ids"])
                if self.config.train.eval_batch_size <= num_eval_data:
                    batch_size = self.config.train.eval_batch_size
                else:
                    batch_size = num_eval_data
                valid_loader = self.per_epoch_dataloader(valid_data_pt,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         intra_domain_shuffle=False,
                                                         drop_last=False)
                val_loss, val_err = self.evaluate(self.model, valid_loader)
                elapsed = time.time() - epoch_start_time
                self.log("-" * 89)
                self.log(
                    f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                    f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
                )
                self.log("-" * 89)
                if val_loss < best_val_loss:
                    self.update_best_model(val_loss, epoch)
            self.lr_schedulers_step()

    def update_best_model(self, val_loss, epoch):
        self.best_model = copy.deepcopy(self.model)
        self.best_model_epoch = epoch
        self.log(f"Best model with score {val_loss:5.4f}")
