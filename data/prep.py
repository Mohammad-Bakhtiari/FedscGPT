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


def plot_umap(adata):
    """
    Compute (if needed) and display basic info about UMAP.
    Stores UMAP in adata.obsm['X_umap'].
    """
    print(f"    n_obs = {adata.n_obs}, n_vars = {adata.n_vars}")
    print("Batch value counts:\n", adata.obs["batch"].value_counts(), "\n")

    # 1) Compute PCA if missing
    if "X_pca" not in adata.obsm_keys():
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
    batch_key: str = "batch",
    query_batch: int = 2,
):
    """
    Split `adata` into:
      • query subset where adata.obs[batch_key] == query_batch
      • reference subset where adata.obs[batch_key] != query_batch

    Saves each subset to its respective .h5ad file.
    """
    if batch_key not in adata.obs_keys():
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs.")

    query_mask = adata.obs[batch_key] == query_batch
    query = adata[query_mask].copy()
    reference = adata[~query_mask].copy()
    for layer_key, arr in adata.obsm.items():
        if arr.shape[0] == adata.n_obs:
            query.obsm[layer_key] = arr[mask.values, :].copy()
            reference.obsm[layer_key] = arr[(~mask.values), :].copy()
    if query.n_obs == 0:
        sys.stderr.write(f"⚠️ Warning: query is empty (no cells with {batch_key} == {query_batch}).\n")
    if reference.n_obs == 0:
        sys.stderr.write(f"⚠️ Warning: reference is empty (no cells with {batch_key} != {query_batch}).\n")

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


def consistent_cell_types(adata: anndata.AnnData, celltype_key: str):
    """
    Ensure that `adata.obs[celltype_key]` is categorical and that all categories
    present in the dataset are explicitly set as categories.

    Args:
        adata: AnnData object.
        celltype_key: column in adata.obs that holds cell‐type labels.
    """
    if celltype_key not in adata.obs_keys():
        raise KeyError(f"Cell type key '{celltype_key}' not found in adata.obs.")

    # Convert to string first, then to categorical
    adata.obs[celltype_key] = adata.obs[celltype_key].astype(str).astype("category")

    # Explicitly set categories to all unique values by reassigning the result
    unique_cts = adata.obs[celltype_key].cat.categories.tolist()
    adata.obs[celltype_key] = adata.obs[celltype_key].cat.set_categories(unique_cts)

    print(f"'{celltype_key}' set as categorical with categories: {unique_cts}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CellLine dataset for benchmarking.")

    parser.add_argument(
        "--orig_path",
        type=str,
        required=True,
        help="Path to the original CellLine AnnData file (e.g., CellLine.h5ad).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the processed data (reference/query).",
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        default="reference_with_umap.h5ad",
        help="Filename for the reference subset (saved under --output_dir).",
    )
    parser.add_argument(
        "--query_file",
        type=str,
        default="query_with_umap.h5ad",
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
    parser.add_argument(
        "--query_batch",
        type=int,
        default=2,
        help="Which batch value to use for the query subset.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If set, normalize adata.X before splitting (using --norm_method).",
    )
    parser.add_argument(
        "--norm_method",
        type=str,
        default="min_max",
        choices=["log", "quantile", "z_score", "min_max"],
        help="Normalization method to apply if --normalize is set.",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load the original AnnData
    print(f"Loading AnnData from: {args.orig_path}")
    adata = anndata.read_h5ad(args.orig_path)
    print(f"    n_obs = {adata.n_obs}, n_vars = {adata.n_vars}\n")

    # 2) Ensure cell types are categorical (consistent across all cells)
    consistent_cell_types(adata, args.celltype_key)

    # 3) If requested, normalize the data matrix
    if args.normalize:
        print(f"Normalizing adata.X using method '{args.norm_method}'.")
        adata.X = normalize_data(adata.X, args.norm_method)
        print("Normalization complete.\n")

    # 4) Compute PCA + neighbors + UMAP
    plot_umap(adata)

    # 5) Split into reference / query and save to disk
    reference_out = os.path.join(args.output_dir, args.reference_file)
    query_out = os.path.join(args.output_dir, args.query_file)

    ref_query_split(
        adata,
        reference_out,
        query_out,
        batch_key=args.batch_key,
        query_batch=args.query_batch,
    )

    print("All done.")
