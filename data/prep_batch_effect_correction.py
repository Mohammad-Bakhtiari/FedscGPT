import os
import sys
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import anndata

from sklearn.preprocessing import (
    LabelEncoder,
    QuantileTransformer,
    StandardScaler,
    MinMaxScaler,
)
from scipy import sparse


def calc_umap(adata):
    """
    Compute (if needed) and display basic info about UMAP.
    Stores UMAP in adata.obsm['X_umap'].
    """
    sc.tl.pca(adata, n_comps=20)
    sc.pp.neighbors(adata, use_rep='X_pca', n_pcs=20)
    sc.tl.umap(adata)


def ref_query_split(
    adata: anndata.AnnData,
    reference_out: str,
    query_out: str,
    split_key: str = "batch",
    query_set_vale: int = 2,
    celltype_key: str = "cell_type",
):
    """
    Split `adata` into:
      • query subset where adata.obs[batch_key] == query_batch
      • reference subset where adata.obs[batch_key] != query_batch

    Saves each subset to its respective .h5ad file.
    """
    if split_key not in adata.obs_keys():
        raise KeyError(f"Batch key '{split_key}' not found in adata.obs.")

    query_mask = adata.obs[split_key] == query_set_vale
    query = adata[query_mask].copy()
    reference = adata[~query_mask].copy()
    for layer_key, arr in adata.obsm.items():
        if arr.shape[0] == adata.n_obs:
            query.obsm[layer_key] = arr[query_mask.values, :].copy()
            reference.obsm[layer_key] = arr[(~query_mask.values), :].copy()
    query.var = adata.var.copy()
    reference.var = adata.var.copy()
    unique_cts = adata.obs[celltype_key].cat.categories.tolist()
    query.obs[celltype_key] = pd.Categorical(query.obs[celltype_key], categories=unique_cts)
    reference.obs[celltype_key] = pd.Categorical(reference.obs[celltype_key], categories=unique_cts)
    if query.n_obs == 0:
        sys.stderr.write(f"⚠️ Warning: query is empty (no cells with {split_key} == {query_set_vale}).\n")
    if reference.n_obs == 0:
        sys.stderr.write(f"⚠️ Warning: reference is empty (no cells with {split_key} != {query_set_vale}).\n")

    print(f"Query     : {query.n_obs} cells × {query.n_vars} genes")
    print(f"Reference : {reference.n_obs} cells × {reference.n_vars} genes\n")

    print(f"Writing reference → {reference_out}")
    reference.write_h5ad(reference_out)

    print(f"Writing query     → {query_out}")
    query.write_h5ad(query_out)


def normalize_data(data: np.ndarray or pd.Series, method: str) -> np.ndarray:
    """
    Normalize the expression matrix or a single vector.

    Args:
        data: np.ndarray of shape (n_cells, n_genes) or pd.Series of length n_cells.
        method: one of {"log", "quantile", "z_score", "min_max"}.

    Returns:
        A numpy array of the normalized data.
    """
    data = data.A if sparse.issparse(data) else data
    if isinstance(data, pd.Series):
        data = data.to_numpy().reshape(-1, 1)

    if method == "log":
        return np.log1p(data)
    elif method == "quantile":
        transformer = QuantileTransformer(output_distribution="uniform")
        return transformer.fit_transform(data)
    elif method == "z_score":
        scaler = StandardScaler()
        return scaler.fit_transform(data)
    elif method == "min_max":
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")



DROPPED_CELLTYPES = {"covid": ['Signaling Alveolar Epithelial Type 2', 'IGSF21+ Dendritic', 'Megakaryocytes'],
                     }
Batch_Mapping = {
    "covid": {
        "Krasnow_distal 1a": "Krasnow",
        "Krasnow_distal 2": "Krasnow",
        "Krasnow_distal 3": "Krasnow",
        "Krasnow_medial 2": "Krasnow",
        "Krasnow_proximal 3": "Krasnow",
        "Sun_sample1_CS": "Sun",
        "Sun_sample2_KC": "Sun",
        "Sun_sample3_TB": "Sun",
        "Sun_sample4_TC": "Sun",
        "Oetjen_U": "Oetjen",
        "Oetjen_P": "Oetjen",
        "Oetjen_A": "Oetjen",
    }
}
QUERY_BATCHES = {"covid": ['Krasnow', 'Sun', 'Freytag'],
                 }

def preprocess(dataset, raw_data_path, batch_key, celltype_key):
    adata = anndata.read_h5ad(raw_data_path)
    adata.obs['batch_group'] = adata.obs[batch_key].replace(Batch_Mapping[dataset])
    adata = adata[~adata.obs[celltype_key].isin(DROPPED_CELLTYPES[dataset])].copy()
    adata.obs["ref-query-split"] = adata.obs["batch_group"].apply(lambda x: "q" if x in QUERY_BATCHES[dataset] else "ref")
    return adata

def preprocess_for_batch_effect_correction(dataset, raw_data_path, prep_for_be_datapath,  reference_file, query_file, batch_key, celltype_key):
    adata = preprocess(dataset, raw_data_path, batch_key, celltype_key)
    raw = adata.copy()
    raw.x = normalize_data(raw, 'min_max')
    calc_umap(raw)
    ref_query_split(raw, reference_file, query_file, split_key="ref-query-split", query_set_vale="q", celltype_key=celltype_key)
    calc_umap(adata)
    adata.X = normalize_data(adata.X, "log")
    adata.write_h5ad(prep_for_be_datapath)

def postprocess_corrected_data(corrected_data_path, reference_file, query_file, celltype_key="cell_type"):
    adata = anndata.read_h5ad(corrected_data_path)
    adata.X = normalize_data(adata.X, "min_max")
    calc_umap(adata)
    ref_query_split(
        adata,
        reference_file,
        query_file,
        split_key="ref-query-split",
        query_set_vale="q",
        celltype_key=celltype_key
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CellLine dataset for benchmarking.")
    parser.add_argument("--prep_type", type=str, default="pre", help="Stage of the pipeline to run.", choices=["pre", "post"])
    parser.add_argument("--dataset", type=str, default="covid", help="Dataset name.", choices=["covid"])
    parser.add_argument("--raw_data_path", type=str, help="Raw data path.")
    parser.add_argument("--prep_for_be_datapath", type=str, help="Path to save pre-processed data for batch effect correction.")
    parser.add_argument("--corrected_data_path", type=str, help="Corrected data path.")
    parser.add_argument("--reference_file",        type=str,        help="Path to the reference adata file.",    )
    parser.add_argument("--query_file",        type=str,        help="Path to the query adata file.",    )
    parser.add_argument("--celltype_key",        type=str,        default="cell_type",        help="Column in adata.obs that holds true cell‐type labels.",    )
    parser.add_argument("--batch_key",        type=str,        default="batch",        help="Column in adata.obs that holds batch labels.",    )
    args = parser.parse_args()
    if args.prep_type == "pre":
        preprocess_for_batch_effect_correction(
            dataset=args.dataset,
            raw_data_path=args.raw_data_path,
            prep_for_be_datapath=args.prep_for_be_datapath,
            batch_key=args.batch_key,
            celltype_key=args.celltype_key,
            reference_file = args.reference_file,
            query_file = args.query_file,
        )
    elif args.prep_type == "post":
        postprocess_corrected_data(
            corrected_data_path=args.corrected_data_path,
            reference_file=args.reference_file,
            query_file=args.query_file,
            celltype_key=args.celltype_key
        )
    else:
        raise ValueError(f"Unknown prep_type: {args.prep_type}")



