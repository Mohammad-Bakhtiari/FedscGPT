import copy
import os.path
import torch
from typing import Dict
from FedscGPT.base import FedBase, BaseClientMixin
from FedscGPT.centralized.annotator import Training, Inference
from FedscGPT.utils import read_h5ad
from FedscGPT.preprocessor.local import Preprocessor
from FedscGPT.preprocessor.aggregation import aggregate_gene_counts, aggregate_bin_edges, aggregate_hvg_stats, \
    aggregate_local_gene_sets, aggregate_local_celltype_sets
from FedscGPT.federated.aggregator import FedAvg


class ClientAnnotator(BaseClientMixin, Training):
    """
    cell_id2type: Here is calculated locally. No global ID!
    """

    def __init__(self, **kwargs):
        Training.__init__(self, **kwargs)
        self.n_samples = len(self.adata)
        self.preprocessor = Preprocessor(
            log=self.log,
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

    def get_local_gene_set(self):
        return self.preprocessor.get_local_gene_set(self.adata)

    def get_local_celltype_set(self):
        return self.preprocessor.get_local_celltype_set(self.adata, self.celltype_key)

    def check_local_gene_set(self, global_gene_dict: Dict[int, str]):
        assert set(self.adata.var.index.tolist()) == set(
            global_gene_dict.values()), "Local gene set is not consistent with global gene set."

    def fed_harmonize(self, global_cellytpe_dict: Dict[int, str]):
        """
        In centralized all unique cell types can be read from the dataset.
        Returns
        -------

        """
        self.adata.obs["batch_id"] = self.adata.obs[self.batch_key].astype("category").cat.codes.values
        cellytpe_dict = {v: k for k, v in global_cellytpe_dict.items()}
        self.adata.obs["celltype_id"] = [cellytpe_dict[i] for i in self.adata.obs[self.celltype_key]]
        self.config.model.n_cls = len(global_cellytpe_dict) if self.config.train.CLS else 1
        self.cell_id2type = global_cellytpe_dict
        self.adata.var["gene_name"] = self.adata.var.index.tolist()
        self.unique_cell_types = list(global_cellytpe_dict.values())
        self.load_pretrained_config()
        self.filter(self.adata)

    def local_harmonize(self):
        super().harmonize(self.adata)

    def get_local_gene_counts(self):
        return self.preprocessor.local_gene_counts(self.adata)

    def apply_gene_mask(self, gene_mask):
        self.adata = self.preprocessor.apply_global_gene_counts(self.adata, gene_mask)

    def filter_cells(self):
        self.adata = self.preprocessor.filter_cells(self.adata)

    def total_normalization(self):
        self.adata = self.preprocessor.total_normalization(self.adata)

    def log1p(self):
        self.adata = self.preprocessor.log1p(self.adata)

    def get_local_hvg_stats(self):
        return self.preprocessor.compute_local_hvg_stats(self.adata)

    def apply_hvg_stats(self, global_hvg_stats):
        self.adata = self.preprocessor.subset_hvgs(self.adata, global_hvg_stats, n_top_genes=None)

    def get_local_bin_edges(self):
        return self.preprocessor.compute_local_bin_edges(self.adata)

    def binning(self, global_bin_edges):
        self.adata = self.preprocessor.apply_binning(self.adata, global_bin_edges)




class FedAnnotator(FedBase, FedAvg):
    def __init__(self, reference_adata, data_dir, output_dir, **kwargs):
        FedBase.__init__(self, data_dir=data_dir, output_dir=output_dir, **kwargs)
        FedAvg.__init__(self, self.fed_config.n_rounds)
        adata = read_h5ad(data_dir, reference_adata)
        self.distribute_adata_by_batch(adata, kwargs['batch_key'])
        for c in range(self.n_clients):
            client = ClientAnnotator(reference_adata='adata.h5ad',
                                     data_dir=self.clients_data_dir[c],
                                     output_dir=self.clients_output_dir[c],
                                     log_id=f"client_{self.client_ids[c]}",
                                     logger=self.logger, **kwargs)
            self.clients.append(client)
        self.retain_best_model_retain(False)

    def aggregate_gene_sets(self):
        local_gene_sets = [client.get_local_gene_set() for client in self.clients]
        global_gene_dict = aggregate_local_gene_sets(local_gene_sets)
        for client in self.clients:
            client.check_local_gene_set(global_gene_dict)

    def aggregate_celltype_sets(self):
        local_celltype_sets = [client.get_local_celltype_set() for client in self.clients]
        global_celltype_dict = aggregate_local_celltype_sets(local_celltype_sets)
        for client in self.clients:
            client.fed_harmonize(global_celltype_dict)

    def load_pretrained_config(self):
        for client in self.clients:
            client.load_pretrained_config()
    def filter_genes(self):
        for client in self.clients:
            client.adata = client.filter(client.adata)

    def preprocess_data(self):
        if self.fed_config.preprocess.filter_gene_by_counts:
            self.logger.federated("Federated filtering genes by counts ...")
            local_gene_counts_list = [client.get_local_gene_counts() for client in self.clients]
            global_gene_mask = aggregate_gene_counts(self.fed_config.preprocess.filter_gene_by_counts,
                                                     local_gene_counts_list)
            for client in self.clients:
                client.apply_global_gene_counts(global_gene_mask)
        if self.fed_config.preprocess.filter_cell_by_counts:
            self.logger.federated("Local filtering cells by counts ...")
            for client in self.clients:
                client.filter_cells()
        if self.fed_config.preprocess.normalize_total:
            self.logger.federated("Local normalization of total counts ...")
            for client in self.clients:
                client.total_normalization()
        if self.fed_config.preprocess.log1p:
            self.logger.federated("Local log1p transformation ...")
            for client in self.clients:
                client.log1p()
        if self.fed_config.preprocess.subset_hvg:
            self.logger.federated("Federated subset HVGs ...")
            # TODO: Validate the code
            local_hvg_stats = [client.get_local_hvg_stats() for client in self.clients]
            global_hvg_stats = aggregate_hvg_stats(local_hvg_stats)
            for client in self.clients:
                client.apply_hvg_stats(global_hvg_stats)
        if self.fed_config.preprocess.binning:
            self.logger.federated("Federated binning ...")
            local_bin_edges_list = [client.get_local_bin_edges() for client in self.clients]
            global_bin_edges = aggregate_bin_edges(local_bin_edges_list)
            for client in self.clients:
                client.binning(global_bin_edges)

    def post_prep_setup(self):
        for client in self.clients:
            self.logger.federated(f"Setting up client {client.log_id} ...")
            client.post_prep()
            client.tokenize()
            client.instantiate_transformer_model()
            client.load_pretrained_model()
            client.setup_losses()