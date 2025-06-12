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


def calc_umap(adata, overwrite=False):
    """
    Compute (if needed) and display basic info about UMAP.
    Stores UMAP in adata.obsm['X_umap'].
    """
    if "X_umap" not in adata.obsm_keys() or overwrite:
        # 1) Compute PCA if missing
        if "X_pca" not in adata.obsm_keys() or overwrite:
            print("X_pca not found → Running PCA (n_comps=50).")
            sc.pp.pca(adata, n_comps=50, svd_solver="arpack")
        else:
            print("X_pca already exists; skipping PCA.")

        # 2) Build neighbors graph (using X_pca)
        print("Building neighbors graph (n_neighbors=30, use_rep='X_pca').")
        sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_pca")

        # 3) Compute UMAP
        print("Computing UMAP (min_dist=0.5).")
        sc.tl.umap(adata, min_dist=0.5)
        print("UMAP stored in adata.obsm['X_umap'].\n")


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
    query.obs[celltype_key] = query.obs[celltype_key].cat.set_categories(unique_cts)
    reference.obs[celltype_key] = reference.obs[celltype_key].cat.set_categories(unique_cts)
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


def combine_covid_batches(adata, batch_key="batch", new_batch_column_name='batch_group'):
    """
    Combine fine‐grained study labels into broader batch groups.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data object containing per‐cell metadata in `adata.obs`.

    Returns
    -------
    adata : anndata.AnnData
        The modified AnnData object with a new column `batch_group` in `adata.obs`.
    Notes
    -----
    - Merges multiple small sub‐studies from the same lab/protocol (e.g. Krasnow slices,
      Sun donors, Oetjen lobes) into single labels (“Krasnow”, “Sun”, “Oetjen”) to:
        * Increase sample size per batch for more stable batch‐effect estimation.
        * Avoid overfitting correction parameters on tiny batches.
        * Preserve major technical differences across labs/platforms.
    """
    mapping = {
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
    adata.obs[new_batch_column_name] = adata.obs[batch_key].replace(mapping)
    unique_batches = adata.obs[new_batch_column_name].unique()
    adata.obs[new_batch_column_name] = adata.obs[new_batch_column_name].cat.set_categories(unique_batches)
    return adata


def detect_standalone_celltypes(adata, celltype_key, batch_key):
    """
    Identify cell types that appear in only one batch, record and print their batch.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell data with .obs containing cell type and batch annotations.
    celltype_key : str
        Column name in adata.obs for cell type labels.
    batch_key : str
        Column name in adata.obs for batch labels.

    Returns
    -------
    dict
        Mapping of each cell type present in exactly one batch to that batch label.
    """
    # DataFrame view
    df = adata.obs[[celltype_key, batch_key]].copy()

    # Count unique batches per cell type
    batch_counts = df.groupby(celltype_key)[batch_key].nunique()

    # Find standalone cell types
    standalone_cts = batch_counts[batch_counts == 1].index.tolist()

    # Build mapping
    mapping = {}
    for ct in standalone_cts:
        batch_val = df.loc[df[celltype_key] == ct, batch_key].iloc[0]
        mapping[ct] = batch_val

    # Print results
    if mapping:
        print(f"Detected {len(mapping)} standalone cell types:")
        for ct, batch in mapping.items():
            print(f"  Cell type '{ct}' appears only in batch '{batch}'")
    else:
        print("No standalone cell types found.")

    # Store mapping
    adata.uns['standalone_celltypes'] = mapping

    return adata



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CellLine dataset for benchmarking.")
    parser.add_argument("data_dir", type=str, help="Directory containing the dataset files.")
    parser.add_argument(
        "--orig_adata",
        type=str,
        required=True,
        help="Path to the original CellLine AnnData file (e.g., CellLine.h5ad).",
    )
    parser.add_argument("--uncorrected_adata", type=str, help="Uncorrected AnnData file name.")
    parser.add_argument("--corrected_adata", type=str, help="Corrected AnnData file name.")
    parser.add_argument(
        "--reference_file",
        type=str,
        default="reference.h5ad",
        help="Filename for the reference subset (saved under --output_dir).",
    )
    parser.add_argument(
        "--query_file",
        type=str,
        default="query.h5ad",
        help="Filename for the query subset (saved under --output_dir).",
    )
    parser.add_argument(
        "--celltype_key",
        type=str,
        default="cell_type",
        help="Column in adata.obs that holds true cell‐type labels.",
    )
    parser.add_argument(
        "--batch_key",
        type=str,
        default="batch",
        help="Column in adata.obs that holds batch labels.",
    )
    parser.add_argument("--ref-query_split_key", type=str, default="ref-query-split", help="Key to split reference and query subsets.")
    parser.add_argument("--batch_group_key", type=str, default="batch_group", help="Column in adata.obs that holds batch group labels.")
    parser.add_argument(
        "--query_set_value",
        type=str,
        default='q',
        help="Which batch value to use for the query subset.",
    )
    parser.add_argument("--stage", type=str, choices=["prep_uncorrected", "split"])


    args = parser.parse_args()
    if args.stage == "prep_uncorrected":
        adata = anndata.read_h5ad(os.path.join(args.data_dir, args.orig_adata))
        adata.X = normalize_data(adata.X, "log")
        print("Normalization complete.\n")

        adata = combine_covid_batches(adata, batch_key=args.batch_key, new_batch_column_name=args.batch_group_key)
        adata = detect_standalone_celltypes(adata, args.celltype_key, args.batch_group_key)
        adata.write_h5ad(os.path.join(args.data_dir, args.uncorrected_adata))
    elif args.stage == "split":
        uncorrected_adata = anndata.read_h5ad(os.path.join(args.data_dir, args.uncorrected_adata))
        corrected_adata = anndata.read_h5ad(os.path.join(args.data_dir, args.corrected_adata))
        standalone_ct = uncorrected_adata.uns["standalone_celltypes"].keys()
        standalone_adata = uncorrected_adata[uncorrected_adata.obs[args.celltype_key].isin(standalone_ct)].copy()
        adata = anndata.concat([corrected_adata, standalone_adata], join='outer')
        adata.uns = uncorrected_adata.uns.copy()
        adata.var = uncorrected_adata.var.copy()
        data.X = normalize_data(adata.X, "min_max")
        calc_umap(adata, overwrite=True)
        if type(args.query_set_value) != str:
            args.query_set_value = str(args.query_set_value)
        ref_query_split(
            adata,
            os.path.join(args.data_dir, args.reference_file),
            os.path.join(args.data_dir, args.query_file),
            split_key=args.ref_query_split_key,
            query_set_vale=args.query_set_value,
            celltype_key=args.celltype_key
        )





