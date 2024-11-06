import time
import anndata
import scanpy as sc
import torch
import numpy as np
import pandas as pd
from gears import PertData
from scgpt.utils import map_raw_id_to_vocab_id
import os
import pickle
from gears.inference import deeper_analysis, non_dropout_analysis
from torch_geometric.loader import DataLoader
from gears.utils import create_cell_graph_dataset_for_prediction
from typing import Iterable, List, Tuple, Dict, Union, Optional
from scgpt.loss import masked_relative_error
import itertools
from tqdm import tqdm
import faiss
from FedscGPT.centralized.models import ScGPT
from FedscGPT.utils import SEED, compute_perturbation_metrics, plot_condition_heatmap, TopKMetricsEvaluator


class Base(ScGPT):
    def __init__(self, dataset_name, pyg_path, split_path, split, reverse, pool_size, early_stop, **kwargs):
        super().__init__(**kwargs)
        self.config.train.early_stop = early_stop
        self.config.log.pool_size = pool_size
        split_path = f"{split_path}/{dataset_name}_{split if reverse else 'simulation'}_{SEED}_0.75.pkl"
        self.pert_data = ClientPertData(self.data_dir, dataset_name, pyg_path, split_path, split, self.verbose, self.log)
        self.adata = None
        self.gene_ids = None
        self.genes = None
        self.n_genes = None
        self.model = None
        self.reverse = reverse


    def setup(self):
        self.load_pretrained_config()
        self.vocab.set_default_index(self.vocab["<pad>"])
        self.genes = self.adata.var["gene_name"].tolist()
        default = self.vocab[self.config.preprocess.pad_token]
        self.gene_ids = np.array([self.vocab[gene] if gene in self.vocab else default for gene in self.genes],
                                 dtype=int)
        self.n_genes = len(self.genes)
        self.instantiate_transformer_model(foundation_type="generator")
        self.load_pretrained_model()

    def load_pretrained_config(self):
        model_configs = super().load_pretrained_config()
        self.config.model.embsize = model_configs["embsize"]
        self.config.model.nhead = model_configs["nheads"]
        self.config.model.d_hid = model_configs["d_hid"]
        self.config.model.nlayers = model_configs["nlayers"]
        self.config.model.nlayers_cls = model_configs["n_layers_cls"]

    def evaluate(self, model, loader: DataLoader, return_raw: bool = False) -> dict:
        """
        Evaluate the model on a given evaluation dataset to measure its performance.
        This method performs a forward pass on the evaluation data to generate predictions for all genes.
        It also specifically extracts predictions for differentially expressed genes.
        Parameters
        ----------
        model : torch.nn.Module
            The model to be evaluated.
        loader : DataLoader
            DataLoader providing the evaluation dataset in batches.
        return_raw : bool, optional
            Whether to return raw results without any processing (default is False).

        Returns
        -------
        dict
            A dictionary containing the evaluation results with the following keys:
            - "pert_cat": ndarray
                An array of perturbation categories.
            - "pred": ndarray
                An array of predicted values for all genes across all batches.
            - "truth": ndarray
                An array of true labels for all genes across all batches.
            - "pred_de": ndarray
                An array of predicted values for differentially expressed genes.
            - "truth_de": ndarray
                An array of true labels for differentially expressed genes.
        """
        if not self.reverse:
            return self.eval_perturb(model, loader, return_raw)
        model.eval()
        total_loss = 0.0
        total_error = 0.0
        n_val_samples = 0
        with (torch.no_grad()):
            for batch_data in loader:
                x, ori_gene_values, pert_flags, target_gene_values = self.unwrap_batch_data(batch_data)
                if self.config.preprocess.include_zero_gene in ["all", "batch-wise"]:
                    input_pert_flags, input_values, mapped_input_gene_ids, src_key_padding_mask, target_values = self.filter_and_sample_genes(
                        ori_gene_values, pert_flags, target_gene_values)
                else:
                    raise ValueError(f"Invalid value for include_zero_gene: {self.config.preprocess.include_zero_gene}")
                with torch.cuda.amp.autocast(enabled=self.config.train.amp):
                    output_dict = self.model(
                        mapped_input_gene_ids,
                        input_values,
                        input_pert_flags,
                        src_key_padding_mask=src_key_padding_mask,
                        **self.train_kwarg
                    )
                    # Different mask for perturbation:  generates a mask where all positions are True, regardless of the values in input_values
                    masked_positions = torch.ones_like(input_values, dtype=torch.bool)
                    loss = self.losses['mse'](output_dict["mlm_output"], target_values, masked_positions)
                    n_val_samples += len(batch_data)
                total_loss += loss.item()
                total_error += masked_relative_error(output_dict["mlm_output"], target_values, masked_positions).item()
            return total_loss / n_val_samples, total_error / n_val_samples

    def eval_perturb(self, model, loader: DataLoader, return_raw: bool = False) -> dict:
        model.eval()
        predictions = []
        true_labels = []
        pert_cat = []
        pred_de = []
        truth_de = []
        with (torch.no_grad()):
            for batch_data in loader:
                # Different args for model ==> no unwrap batch data
                # no src_key_padding_mask needed
                # No auto-casting for evaluation
                pert_cat.extend(batch_data.pert)
                p = model.pred_perturb(
                    batch_data,
                    include_zero_gene=self.config.preprocess.include_zero_gene,
                    gene_ids=self.gene_ids,
                )
                predictions.extend(p.cpu())
                true_labels.extend(batch_data.y.cpu())

                # Differentially expressed genes
                for itr, de_idx in enumerate(batch_data.de_idx):
                    pred_de.append(p[itr, de_idx])
                    truth_de.append(batch_data.y[itr, de_idx])

                # No loss evaluation for perturbation

        # all genes
        results = {
            "pert_cat": np.array(pert_cat),
            "pred": torch.stack(predictions).detach().cpu().numpy().astype(float),
            "truth": torch.stack(true_labels).detach().cpu().numpy().astype(float),
            "pred_de": torch.stack(pred_de).detach().cpu().numpy().astype(float),
            "truth_de": torch.stack(truth_de).detach().cpu().numpy().astype(float),
        }
        return results

    def exclude_ctrl_cond(self):
        """ exclude ctrl condition from cell graph of training and validation dataloaders

        """
        if "ctrl" in self.pert_data.set2conditions['train']:
            self.pert_data.set2conditions['train'].remove("ctrl")
            if self.verbose:
                self.log("Removed ctrl from training cell graph")
        if "ctrl" in self.pert_data.set2conditions['val']:
            self.pert_data.set2conditions['val'].remove("ctrl")
            if self.verbose:
                self.log("Removed ctrl from validation cell graph")


    def drop_non_local_conditions(self):
        """ Drop non-local conditions from the set2conditions dictionary

        """
        unique_local_cond = self.adata.condition.unique()
        default_conditions = self.pert_data.set2conditions.keys()
        for cond in default_conditions:
            if cond not in unique_local_cond:
                return self.pert_data.set2conditions.remove(cond)
        if self.verbose:
            self.log(f"Non-local conditions dropped from set2conditions: {set(default_conditions) - set(unique_local_cond)}")



class Training(Base):
    def __init__(self, reference_adata, **kwargs):
        super().__init__(split="no_test", **kwargs)
        if not os.path.isabs(reference_adata):
            reference_adata = os.path.join(self.data_dir, reference_adata)
        self.pert_data.load(data_path=reference_adata)
        if self.verbose:
            self.log(f"Data loaded from {reference_adata}")
        self.exclude_ctrl_cond()
        self.pert_data.get_dataloader(self.config.train.batch_size)
        self.adata = self.pert_data.adata

    def train_and_validate(self):
        epoch_start_time = time.time()
        best_val_corr, patience = 0, 0
        for epoch, val_res in enumerate(self.train(), 1):
            # Per epoch post evaluate validation
            if self.reverse:
                val_loss, val_mre = val_res
                elapsed = time.time() - epoch_start_time
                self.log(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss/mse {val_loss:5.4f} |")
                val_criteria = val_loss
            else:
                val_metrics = compute_perturbation_metrics(val_res, self.adata[self.adata.obs["condition"] == "ctrl"])
                elapsed = time.time() - epoch_start_time
                self.log(f"val_metrics at epoch {epoch}: {val_metrics}")
                self.log(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")
                val_criteria = val_metrics["pearson"]
            if val_criteria > best_val_corr:
                best_val_corr = val_criteria
                self.update_best_model(best_val_corr, epoch)
                patience = 0
            elif self.config.train.early_stop:
                patience += 1
                if patience >= self.config.train.early_stop:
                    self.log(f"Early stop at epoch {epoch}")
                    break
            self.lr_schedulers_step()
            epoch_start_time = time.time()

    def get_train_valid_loaders(self, epoch):
        train_loader = self.pert_data.dataloader["train_loader"]
        valid_loader = self.pert_data.dataloader["val_loader"]
        return train_loader, valid_loader


    def train_on_batch(self, batch_data):
        x, ori_gene_values, pert_flags, target_gene_values = self.unwrap_batch_data(batch_data)
        if self.config.preprocess.include_zero_gene in ["all", "batch-wise"]:
            input_pert_flags, input_values, mapped_input_gene_ids, src_key_padding_mask, target_values = self.filter_and_sample_genes(
                ori_gene_values, pert_flags, target_gene_values)
        else:
            raise ValueError(f"Invalid value for include_zero_gene: {self.config.preprocess.include_zero_gene}")
        with torch.cuda.amp.autocast(enabled=self.config.train.amp):
            output_dict = self.model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                **self.train_kwarg
            )
            # Different mask for perturbation:  generates a mask where all positions are True, regardless of the values in input_values
            masked_positions = torch.ones_like(input_values, dtype=torch.bool)
            # Different loss (args) for perturbation
            args_dict = {"masked_positions": masked_positions,
                         "target_values": target_values,
                         **output_dict}
            self.apply_loss(**args_dict)
            self.fedprox()
        self.apply_gradients_and_optimize()
        self.loss_meter.reset_batch_loss()



    def filter_and_sample_genes(self, ori_gene_values, pert_flags, target_gene_values):
        """ Filters and samples gene indices based on specific criteria and prepares them for model input.

            This method performs the following operations:
            1. **Filter Gene Indices**: Depending on the configuration, the method either includes all genes or filters to include only non-zero gene values.
               - If `self.config.preprocess.include_zero_gene` is set to "all", all gene indices are included.
               - Otherwise, only the indices of non-zero gene values are retained.

            2. **Sample Gene Subset**: If the number of selected gene indices exceeds the maximum allowed sequence length (`self.config.preprocess.max_seq_len`),
               a random subset of genes is sampled to match this limit.

            3. **Map Gene IDs**: The selected gene indices are then mapped to vocabulary IDs using the `map_raw_id_to_vocab_id` function.

            4. **Prepare Input Tensors**: The method prepares tensors for input values, perturbation flags, and target values, all corresponding to the selected gene indices.
               It also generates a padding mask (`src_key_padding_mask`) that identifies the padded positions in the input sequence.

            Parameters
            ----------
            ori_gene_values : torch.Tensor
                A tensor containing the original gene expression values for each gene in the dataset.

            pert_flags : torch.Tensor
                A tensor indicating the perturbation status for each gene.

            target_gene_values : torch.Tensor
                A tensor containing the target gene expression values that the model aims to predict.

            Returns
            -------
            input_pert_flags : torch.Tensor
                The perturbation flags for the selected subset of genes.

            input_values : torch.Tensor
                The gene expression values for the selected subset of genes.

            mapped_input_gene_ids : torch.Tensor
                The vocabulary-mapped gene IDs for the selected subset of genes, repeated for each batch.

            src_key_padding_mask : torch.Tensor
                A boolean tensor indicating which positions in the input sequence are padding and should be ignored by the model.

            target_values : torch.Tensor
                The target gene expression values for the selected subset of genes.
        """
        batch_size = len(ori_gene_values)
        if self.config.preprocess.include_zero_gene == "all":
            input_gene_ids = torch.arange(self.n_genes, device=self.device, dtype=torch.long)
        else:
            input_gene_ids = (
                ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
            )
        # sample input_gene_id
        if len(input_gene_ids) > self.config.preprocess.max_seq_len:
            input_gene_ids = torch.randperm(len(input_gene_ids), device=self.device)[
                             :self.config.preprocess.max_seq_len
                             ]
        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]
        target_values = target_gene_values[:, input_gene_ids]
        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, self.gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
        # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
        src_key_padding_mask = torch.zeros_like(
            input_values, dtype=torch.bool, device=self.device
        )
        return input_pert_flags, input_values, mapped_input_gene_ids, src_key_padding_mask, target_values

    def unwrap_batch_data(self, batch_data):
        batch_size = len(batch_data.y)
        x: torch.Tensor = batch_data.x.to(self.device)
        ori_gene_values = x[:, 0].view(batch_size, self.n_genes).to(self.device)
        pert_flags = x[:, 1].long().view(batch_size, self.n_genes).to(self.device)
        target_gene_values = batch_data.y.to(self.device)
        return x, ori_gene_values, pert_flags, target_gene_values





class Inference(Base):
    def __init__(self, mode, query_adata, reverse, pyg_path, split_path, **kwargs):
        # split = "no_test" if kwargs['reverse'] else "simulation"
        if reverse:
            assert '/reverse' in pyg_path, "In reverse perturbation scenario, the pyg_path should contain '/reverse'"
            pyg_path = pyg_path.replace('/reverse', '')
            split_path = split_path.replace('/reverse', '')
        super().__init__(reverse=reverse, split='simulation', pyg_path=pyg_path, split_path=split_path, **kwargs)
        self.mode = mode
        # self.pert_data.load(data_path=os.path.join(self.data_dir, os.path.basename(kwargs["reference_adata"])))
        self.pert_data.load(data_path=query_adata)
        self.pert_data.get_dataloader(self.config.train.eval_batch_size)
        self.adata = self.pert_data.adata

        self.gene_raw2id, self.cond2name = None, None
        self.ctrl_data = self.get_cond("ctrl")
        self.records = []

    def setup(self):
        super().setup()
        if not self.reverse:
            self.cond2name = dict(self.adata.obs[["condition", "condition_name"]].values)
            self.gene_raw2id = dict(zip(self.adata.var.index.values, self.adata.var.gene_name.values))
        self.model.eval()

    def evaluate_graph_based_gene_interactions(self, pert_list: List[str]) -> Dict:
        """
        Evaluate graph-based gene interactions for a list of perturbations using a pre-trained model.

        This method computes predictions for gene expression changes based on perturbations
        by leveraging a graph-based model. The predictions are made using control cells as a reference,
        and results are returned for each perturbation in the list.

        Parameters
        ----------
        pert_list : List[str], optional
            A list of perturbations (e.g., gene names) to evaluate. Each perturbation can be
            a single gene or a combination of genes (e.g., ['gene1', 'gene2']). If no perturbations
            are provided, an error is raised.

        Returns
        -------
        Dict
            A dictionary where keys are perturbation names (joined by underscores if multiple genes are involved),
            and values are the predicted gene expression changes.

        Raises
        ------
        ValueError
            If any gene in the provided `pert_list` is not present in the gene list of the perturbation graph.

        Notes
        -----
        - The predictions are computed without updating the model (`torch.no_grad()` is used).
        - The control group cells are used as the reference for prediction.
        - This function requires that the model has already been trained and is loaded in `self.model`.
        """
        if self.config.log.pool_size is None:
            self.config.log.pool_size = len(self.ctrl_data.obs)
            self.log(f"Using all control cells for prediction by setting pool_size to {self.config.log.pool_size}")
        gene_list = self.pert_data.gene_names.values.tolist()
        for pert in pert_list:
            for i in pert:
                if i not in gene_list:
                    raise ValueError(
                        "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                    )
        with torch.no_grad():
            results_pred = {}
            for pert in pert_list:
                cell_graphs = create_cell_graph_dataset_for_prediction(
                    pert, self.ctrl_data, gene_list, self.device, num_samples=self.config.log.pool_size
                )
                loader = DataLoader(cell_graphs, batch_size=self.config.train.eval_batch_size, shuffle=False)
                preds = []
                for batch_data in loader:
                    pred_gene_values = self.model.pred_perturb(
                        batch_data, self.config.preprocess.include_zero_gene, gene_ids=self.gene_ids, amp=self.config.train.amp
                    )
                    preds.append(pred_gene_values)
                preds = torch.cat(preds, dim=0)
                results_pred["_".join(pert)] = preds.detach().cpu().numpy()

        return results_pred


    def predict_perturbation_outcome(self, query: str) -> Tuple:
        """ Predict the outcome of a given perturbation query.

        Parameters
        ----------
        query: str
            A string representing the perturbation of interest. It can be a single gene or a combination of genes
            (e.g., "gene1+gene2"). If the perturbation involves a control gene, it will be indicated as "ctrl"
                (e.g., "gene1+ctrl").

        Returns
        -------
        tuple: A tuple containing:
            - genes (List[str]): List of top differentially expressed genes (DE genes)
                                 corresponding to the perturbation.
            - truth (np.ndarray): Ground truth expression values of the DE genes for the queried condition.
            - pred (np.ndarray): Predicted expression values of the DE genes based on graph-based interactions.
            - ctrl_mean: Mean expression values of the DE genes in the control condition.
        """
        # For testing we need all conditions including the ones used in training
        # The evaluate_graph_based_gene_interactions method does not require test data because it uses control cells
        # as the reference for making predictions about gene expression changes. The method leverages a pre-trained
        # model and computes predictions by comparing perturbations to the control group. Since the control group
        # provides the baseline for comparison, and predictions are based on this reference, the presence of test
        # data is unnecessary for the evaluation process. The method's primary goal is to predict expression changes
        # under perturbation conditions, using the control as the sole baseline for comparison, not needing any
        # information from a test set.
        # TODO: Federated version of this method:
        # TODO: aggregation of stats of local control samples
        # TODO: Aggregation of Ground truth and predicted expression values of the local DE genes for the queried condition.
        # TODO: handle privacy issues by communicating the expression differences rather than values
        # Combine train and test data for control samples

        de_idx = [
            self.pert_data.node_map[self.gene_raw2id[i]]
            for i in self.adata.uns["top_non_dropout_de_20"][self.cond2name[query]]
        ]
        genes = [
            self.gene_raw2id[i] for i in self.adata.uns["top_non_dropout_de_20"][self.cond2name[query]]
        ]
        truth = self.adata[self.adata.obs.condition == query].X.toarray()[:, de_idx]
        self.log(f"Evaluation of {truth.shape[0]} {query} perturbed cell {truth.shape} with {self.ctrl_data.shape[0]} control cells")
        if query.split("+")[1] == "ctrl":
            perturbed_gene = query.split("+")[0]
            pred = self.evaluate_graph_based_gene_interactions([[perturbed_gene]])
            pred = pred[perturbed_gene][:, de_idx]
        else:
            perturbed_genes = query.split("+")
            pred = self.evaluate_graph_based_gene_interactions([perturbed_genes])
            pred = pred["_".join(perturbed_genes)][:, de_idx]
        ctrl_mean = np.mean(self.ctrl_data[:, de_idx].X.toarray(), axis=0)
        return genes, truth, pred, ctrl_mean

    def get_cond(self, query):
        new_adata = self.adata[self.adata.obs["condition"] == query].copy()
        new_adata.uns = self.adata.uns.copy()
        new_adata.vars = self.adata.var.copy()
        return new_adata

    def evaluate(self, **kwargs):

        """
            Perform a deeper subgroup analysis on the test data by evaluating the model,
            computing relevant metrics, and analyzing the results for specific subgroups.

            This method evaluates the model on the test data, computes metrics such as
            Pearson correlation (delta) for gene perturbations, and performs a deeper analysis
            on subgroups of the test data. Results for dropout and non-dropout perturbations
            are also calculated.

            Returns
            -------
            dict
                A dictionary containing:
                    - "test_metrics": The overall metrics computed on the test data.
                    - "subgroup_analysis": A dictionary where keys are subgroup names and values are
                      dictionaries containing metric results (both regular and non-dropout metrics) for each subgroup.

            Notes
            -----
            - The subgroup analysis is based on the predefined subgroups from `self.pert_data.subgroup["test_subgroup"]`.
            - Metrics such as Pearson correlation (delta) are used to evaluate the model's performance on perturbation outcomes.

        """
        test_res = super().evaluate(self.model, self.pert_data.dataloader["test_loader"])
        test_metrics = compute_perturbation_metrics(test_res, self.ctrl_data)
        self.log(test_metrics)
        return test_metrics, test_res

    def deeper_analysis(self, test_res):
        deeper_res = deeper_analysis(self.adata, test_res)
        non_dropout_res = non_dropout_analysis(self.adata, test_res)
        return deeper_res, non_dropout_res


    def subgroup_analysis(self, deeper_res, non_dropout_res):
        metrics = ["pearson_delta", "pearson_delta_de"]
        metrics_non_dropout = [
            "pearson_delta_top20_de_non_dropout",
            "pearson_top20_de_non_dropout",
        ]
        # Step 5: Initialize the subgroup analysis dictionary
        subgroup_analysis = {}
        for name in self.adata.uns["subgroups"]["test_subgroup"].keys():
            subgroup_analysis[name] = {m: [] for m in metrics + metrics_non_dropout}
        # Step 6: Populate subgroup analysis with results
        for name, pert_list in self.adata.uns["subgroups"]["test_subgroup"].items():
            for pert in pert_list:
                for m in metrics:
                    subgroup_analysis[name][m].append(deeper_res[pert][m])
                for m in metrics_non_dropout:
                    subgroup_analysis[name][m].append(non_dropout_res[pert][m])
        # Step 7: Log the mean values of the metrics for each subgroup
        for name, result in subgroup_analysis.items():
            for m in result.keys():
                mean_value = np.mean(subgroup_analysis[name][m])
                self.log(f"test_{name}_{m}: {mean_value}")
        return subgroup_analysis


    def record_results(self, exp_params: dict, metrics: dict, analysis: dict):
        for metric, value in metrics.items():
            record = exp_params.copy()
            record.update({'category': 'test_metrics', 'subgroup': '', 'metric': metric, 'value': value})
            self.records.append(record)

        # Add subgroup_analysis to the dataframe
        for subgroup, metrics in analysis.items():
            for metric, values in metrics.items():
                for value in values:
                    record = exp_params.copy()
                    record.update({'category': 'subgroup_analysis', 'subgroup': subgroup, 'metric': metric, 'value': value})
                    self.records.append(record)


    def save_records(self):
        if not self.reverse:
            file_path = f"{self.output_dir}/records.csv"
            write_header = not os.path.exists(file_path) or not self.append
            df = pd.DataFrame(self.records)
            df.to_csv(file_path, mode='a', header=write_header, index=False)
            if self.verbose:
                self.log(f"Records saved at {file_path}")

    def get_test_gene_list(self):
        test_groups = self.adata.uns['subgroups']["test_subgroup"].copy()
        test_gene_list = []
        for i in test_groups.keys():
            for g in test_groups[i]:
                if g.split('+')[0] != 'ctrl':
                    test_gene_list.append(g.split('+')[0])
                if g.split('+')[1] != 'ctrl':
                    test_gene_list.append(g.split('+')[1])
        return list(set(test_gene_list))

    def get_conditions_list(self):
        test_gene_list = self.get_test_gene_list()

        train_condition_list = self.adata.obs[self.adata.obs.split == 'train'].condition.values
        valid_condition_list = self.adata.obs[self.adata.obs.split == 'val'].condition.values
        test_condition_list = self.adata.obs[self.adata.obs.split == 'test'].condition.values
        return test_gene_list, test_condition_list, train_condition_list, valid_condition_list

    def plot_condition_matrix(self, test_gene_list, test_condition_list, train_condition_list, valid_condition_list):
        def update_condition_matrix(df, condition_list, label):
            for i in condition_list:
                if i != 'ctrl':
                    g0 = i.split('+')[0]
                    g1 = i.split('+')[1]
                    if g0 == 'ctrl' and g1 in test_gene_list:
                        df.loc[g1, g1] = label
                    elif g1 == 'ctrl' and g1 in test_gene_list:
                        df.loc[g0, g0] = label
                    elif g0 in test_gene_list and g1 in test_gene_list:
                        df.loc[g0, g1] = label
                        df.loc[g1, g0] = label

        df = pd.DataFrame(np.zeros((len(test_gene_list), len(test_gene_list))), columns=test_gene_list,
                          index=test_gene_list)
        update_condition_matrix(df, train_condition_list, 'Train')
        update_condition_matrix(df, valid_condition_list, 'Valid')
        update_condition_matrix(df, test_condition_list, 'Test')
        df = df.replace({0: 'Unseen'})

        sub_gene_list = list(
            set(df[(df == 'Train').sum(0) > 0].index).intersection(df[(df == 'Test').sum(0) > 0].index))
        sub_test_gene_list = ((df.loc[:, sub_gene_list] == 'Train').sum(0) + (df.loc[:, sub_gene_list] == 'Test').sum(
            0)).sort_values()[-20:].index
        sub_df = df.loc[sub_test_gene_list, sub_test_gene_list]
        df = df.loc[np.sort(sub_df.index), np.sort(sub_df.index)]
        plot_condition_heatmap(df, os.path.join(self.output_dir, f"gene_condition_heatmap.png"))
        if self.verbose:
            self.log(f"Condition matrix heatmap saved at {os.path.join(self.output_dir, 'gene_condition_heatmap.png')}")


    def evaluate_gene_interactions(self, test_gene_list, test_condition_list, exp_params: Dict[str, Union[int, str]]):
        """
        Main function to evaluate graph-based gene interactions and log results.

        Parameters
        ----------
        test_gene_list : list
            List of test genes to evaluate.
        test_condition_list : list
            List of test conditions to evaluate.
        exp_params: dict
            A dictionary containing the experiment parameters.
        """
        pert_list = self._generate_perturbation_list(test_gene_list)
        results_pred = self.evaluate_graph_based_gene_interactions(pert_list)
        self.move_model_to_cpu()
        xb = self._prepare_faiss_input(results_pred)
        index = self._build_faiss_index(xb)
        sub_test_condition_list = self._filter_test_conditions(test_gene_list, test_condition_list)
        xq, ground_truth = self._prepare_query_data(sub_test_condition_list, pert_list)
        evaluator = TopKMetricsEvaluator(
            experiment_name=self.mode,
            experiment_parameters=exp_params,
            output_dir=self.output_dir
        )
        self._perform_faiss_search(index,
                                   xq,
                                   ground_truth,
                                   results_pred,
                                   list(range(1, self.config.log.top_k + 1)),
                                   evaluator)
        evaluator.finalize_metrics()
        evaluator.write_metrics_to_csv()

    @staticmethod
    def _generate_perturbation_list(test_gene_list):
        """
        Generate a list of perturbations from the test gene list.

        Parameters
        ----------
        test_gene_list : list
            List of test genes to use in the combinations.

        Returns
        -------
        pert_list : list
            List of perturbations (gene combinations).
        """
        pert_list = []
        for comb in itertools.combinations(test_gene_list + ['ctrl'], 2):
            if comb[0] == 'ctrl':
                pert_list.append([comb[1]])
            elif comb[1] == 'ctrl':
                pert_list.append([comb[0]])
            else:
                pert_list.append([comb[0], comb[1]])
        return pert_list

    @staticmethod
    def _prepare_faiss_input(results_pred):
        """
        Prepare the input data for Faiss indexing.

        Parameters
        ----------
        results_pred : dict
            Dictionary of predicted results for gene interactions.

        Returns
        -------
        xb : np.ndarray
            The concatenated results prepared for Faiss indexing.
        """
        results_pred_np = [np.expand_dims(results_pred[p], 0) for p in results_pred.keys()]
        results_pred_np = np.concatenate(results_pred_np)
        M = results_pred_np.shape[-1]
        xb = results_pred_np.reshape(-1, M)
        return xb

    @staticmethod
    def _build_faiss_index(xb):
        """
        Build and return a Faiss index from the input data.

        Parameters
        ----------
        xb : np.ndarray
            The input data for Faiss.

        Returns
        -------
        index : faiss.IndexFlatL2
            The Faiss index constructed from the input data.
        """
        d = xb.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        return index

    @staticmethod
    def _filter_test_conditions(test_gene_list, test_condition_list):
        """
        Filter the test conditions based on the test gene list.

        Parameters
        ----------
        test_gene_list : list
            List of test genes.
        test_condition_list : list
            List of all test conditions.

        Returns
        -------
        sub_test_condition_list : list
            List of filtered test conditions.
        """
        sub_test_condition_list = []
        for c in np.unique(test_condition_list):
            g0, g1 = c.split('+')
            if g0 == 'ctrl' and g1 in test_gene_list:
                sub_test_condition_list.append(c)
            elif g1 == 'ctrl' and g0 in test_gene_list:
                sub_test_condition_list.append(c)
            elif g0 in test_gene_list and g1 in test_gene_list:
                sub_test_condition_list.append(c)
        return sub_test_condition_list

    def _prepare_query_data(self, sub_test_condition_list, pert_list):
        """
        Prepare the query data (xq) and ground truth for Faiss search.

        Parameters
        ----------
        sub_test_condition_list : list
            List of filtered test conditions.
        pert_list : list
            List of perturbations for evaluation.

        Returns
        -------
        xq : np.ndarray
            Query data for Faiss search.
        ground_truth : list
            Ground truth labels for Faiss search.
        """
        q_list, ground_truth = [], []
        for c in tqdm(sub_test_condition_list):
            g0, g1 = c.split('+')
            if g0 == 'ctrl':
                temp, temp1 = [g1], [g1]
            elif g1 == 'ctrl':
                temp, temp1 = [g0], [g0]
            else:
                temp, temp1 = [g0, g1], [g1, g0]
            if temp in pert_list or temp1 in pert_list:
                sub = self.adata[self.adata.obs.split == 'test']
                sub = sub[sub.obs.condition == c]
                q_list.append(sub.X.todense())
                if g0 < g1:
                    ground_truth.extend([c] * sub.X.todense().shape[0])
                else:
                    ground_truth.extend(['+'.join([g1, g0])] * sub.X.todense().shape[0])

        xq = np.concatenate(q_list)
        return xq, ground_truth


    def _perform_faiss_search(self, index, xq, ground_truth, results_pred, k_values: List[int], evaluator: TopKMetricsEvaluator):
        """
        Perform the Faiss search and log results for different k-values, using TopKMetricsEvaluator.

        Parameters
        ----------
        index : faiss.IndexFlatL2
            Faiss index used for searching.
        xq : np.ndarray
            Query data for Faiss search.
        ground_truth : list
            Ground truth labels for Faiss search.
        results_pred : dict
            Dictionary of predicted results for gene interactions.
        k_values : List[int]
            List of k values to consider for evaluation.
        evaluator : TopKMetricsEvaluator
            Evaluator object for computing metrics.
        """


        for k in k_values:
            self.log(f'Top {k}')

            # Perform search with Faiss
            D, I = index.search(xq, k)
            condition_search_df = pd.DataFrame(I)
            condition_search_df = self._replace_index_with_conditions(condition_search_df, results_pred)
            condition_search_df['ground_truth'] = ground_truth
            df_aggr = self._aggregate_predictions(condition_search_df, k, ground_truth)

            pred = df_aggr.values[:, :k]
            truth = df_aggr.values[:, -1]

            # Incrementally add the results to the evaluator for the current k
            for p, t in zip(pred, truth):
                evaluator.add_data(list(p), [t])  # Add predictions and true values incrementally
            evaluator.update_metrics_for_k(k)
            self._log_results(pred, truth, k)

    @staticmethod
    def _replace_index_with_conditions(df, results_pred):
        """
        Replace Faiss index with condition names.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing Faiss index results.
        results_pred : dict
            Dictionary of predicted results for gene interactions.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with replaced condition names.
        """
        ind_list, condition_list = [], []
        ind = 0
        for i in results_pred.keys():
            for j in range(results_pred[i].shape[0]):
                ind_list.append(ind)
                condition_list.append(i)
                ind += 1
        index_to_condition = dict(zip(ind_list, condition_list))
        return df.replace(index_to_condition)

    def _aggregate_predictions(self, df, k, ground_truth):
        """
        Aggregate predictions for each ground truth condition.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing predictions and ground truth.
        k : int
            Number of top predictions to consider.
        ground_truth : list
            Ground truth labels.

        Returns
        -------
        df_aggr : pd.DataFrame
            DataFrame containing aggregated predictions.
        """
        aggr_pred, ground_truth_short = [], []
        for i in np.unique(ground_truth):
            values = df[df.ground_truth == i].loc[:, list(range(k))].values.flatten()
            unique, counts = np.unique(values, return_counts=True)
            if len(counts) < k:
                adjusted_k = len(counts)
                if self.verbose:
                    self.log(f"Adjusted k to counts: counts:{counts} < k:{k} for {i}")
                ind = np.argpartition(-counts, kth=adjusted_k)[:adjusted_k]
            else:
                ind = np.argpartition(-counts, kth=k)[:k]
            aggr_pred.append(np.expand_dims(unique[ind], 0))
            ground_truth_short.append(i)
        return pd.DataFrame(np.concatenate(aggr_pred)).assign(ground_truth=ground_truth_short)

    def _log_results(self, pred, truth, k):
        """
        Log the prediction results for both 2/2 and 1/2 matches.

        Parameters
        ----------
        pred : np.ndarray
            Predicted gene combinations.
        truth : np.ndarray
            Ground truth gene combinations.
        k : int
            Number of top predictions to consider.
        """
        count_2_2, count_1_2 = 0, 0
        for i in range(len(truth)):
            g0, g1 = truth[i].split('+')
            truth0, truth1 = '_'.join([g0, g1]), '_'.join([g1, g0])

            # 2/2 matches
            if truth0 in pred[i, :] or truth1 in pred[i, :]:
                count_2_2 += 1

            # 1/2 matches
            found_one = False
            for j in pred[i, :]:
                if not found_one and (g0 in j or g1 in j):
                    found_one = True
                    count_1_2 += 1

        self.log(f"Top {k} 2/2 = {count_2_2}")
        self.log(f"Top {k} 1/2 = {count_1_2}")

# TODO: Solve the data leakage issue in creating pyg object!
class ClientPertData(PertData):
    def  __init__(self, data_path, data_name, pyg_path, split_path, split, verbose=False, log=print):
        super().__init__(data_path)
        self.pyg_path = pyg_path
        self.verbose = verbose
        self.log = log
        federated_splits = ['simulation', 'no_test']
        assert split in federated_splits, (
            f"Split {split} is not allowed for perturbation prediction! allowed options are {federated_splits}")
        self.split = split
        if os.path.isfile(split_path):
            if self.verbose:
                self.log(f"Loading split file from {split_path}")
            with open(split_path, "rb") as f:
                self.set2conditions = pickle.load(f)
        else:
            raise FileNotFoundError(f"Split file does not exist at {split_path}!")


    def get_dataloader(self, batch_size, test_batch_size = None):
        dl = super().get_dataloader(batch_size, test_batch_size)
        if self.split == "no_split":
            self.dataloader = dl

    def load(self, data_name = None, data_path = None):
        if data_name is None and data_path.endswith('.h5ad'):
            self.adata = sc.read_h5ad(data_path)
            self.dataset_name = data_path.split('/')[-1][:-5]
            self.dataset_path = "/".join(data_path.split('/')[:-1])
        else:
            super().load(data_name=data_name, data_path=data_path)

        if not os.path.exists(self.pyg_path):
            os.mkdir(self.pyg_path)
        dataset_fname = os.path.join(self.pyg_path, 'cell_graphs.pkl')

        if os.path.isfile(dataset_fname):
            self.log("Local copy of pyg dataset is detected. Loading...")
            with open(dataset_fname, "rb") as f:
                self.dataset_processed = pickle.load(f)
            self.log("Done!")
        else:
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
            self.gene_names = self.adata.var.gene_name
            self.log("Creating pyg object for each cell in the data...")
            self.dataset_processed = self.create_dataset_file()
            self.log("Saving new dataset pyg object at " + dataset_fname)
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))
            self.log("Done!")


def read_train_data(data_path):
    train = sc.read_h5ad(data_path)
    train_code2name = dict(train.obs[["condition", "condition_name"]].values)
    ctrl_adata = train[train.obs["condition"] == "ctrl"]
    condition_adata = train[train.obs["condition"] != "ctrl"]
    return train, ctrl_adata, train_code2name, condition_adata