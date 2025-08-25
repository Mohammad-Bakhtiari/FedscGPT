import numpy as np
from typing import Dict, List, Tuple, Set
import scgpt as scg


def aggregate_gene_counts(filter_gene_by_counts, local_gene_counts_list: List[Dict[str, int]],
                          logger: scg.logger = None) -> np.ndarray:
    all_gene_names = list(local_gene_counts_list[0].keys())
    combined_gene_counts = np.zeros(len(all_gene_names))

    for local_counts in local_gene_counts_list:
        for i, gene in enumerate(all_gene_names):
            combined_gene_counts[i] += local_counts[gene]

    global_gene_mask = combined_gene_counts >= filter_gene_by_counts
    s = np.sum(~global_gene_mask)
    if s > 0:
        msg = f"Filtered out {s} genes that are detected in less than {self.filter_gene_by_counts} counts"
        if logger:
            logger.info(msg)
        else:
            print(msg)

    return global_gene_mask


def aggregate_hvg_stats(local_stats_list: List[Dict]) -> Dict:
    all_means = np.stack([stats['means'] for stats in local_stats_list])
    all_variances = np.stack([stats['variances'] for stats in local_stats_list])
    all_variances_norm = np.stack([stats['variances_norm'] for stats in local_stats_list])

    global_means = np.mean(all_means, axis=0)
    global_variances = np.mean(all_variances, axis=0)
    global_variances_norm = np.mean(all_variances_norm, axis=0)

    return {
        'means': global_means,
        'variances': global_variances,
        'variances_norm': global_variances_norm
    }


def aggregate_bin_edges(local_bin_edges_list: List[Tuple[np.ndarray, int]]) -> np.ndarray:
    total_samples = sum([samples for _, samples in local_bin_edges_list])
    n_bins = len(local_bin_edges_list[0][0])

    if any(len(bin_edges) != n_bins for bin_edges, _ in local_bin_edges_list):
        raise ValueError("All local bin edge lists must have the same number of bins.")

    weighted_bin_edges = np.zeros(n_bins)

    for bin_edges, num_samples in local_bin_edges_list:
        weighted_bin_edges += bin_edges * num_samples

    weighted_bin_edges /= total_samples

    global_bin_edges = np.quantile(weighted_bin_edges, np.linspace(0, 1, n_bins))
    return global_bin_edges


def aggregate_local_gene_sets(local_gene_sets: List[Set[str]]) -> Dict[int, str]:
    combined_gene_set = set()
    for gene_set in local_gene_sets:
        combined_gene_set.update(gene_set)
    return dict(enumerate(combined_gene_set))


def aggregate_local_celltype_sets(local_celltype_sets: List[Set[str]]) -> Dict[int, str]:
    combined_celltype_set = set()
    for celltype_set in local_celltype_sets:
        combined_celltype_set.update(celltype_set)
    return dict(enumerate(combined_celltype_set))


