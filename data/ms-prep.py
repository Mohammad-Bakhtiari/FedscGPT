import scanpy as sc
import pandas as pd
import numpy as np
import os


def prep(reference_path, query_path, output_combined):

    query = sc.read_h5ad(query_path)
    reference = sc.read_h5ad(reference_path)

    # === 2. Combine datasets ===
    adata = reference.concatenate(query, batch_key='split_label', batch_categories=["reference", "query"])
    print(f"Combined shape: {adata.shape}")

    adata.obs[disease_key] = adata.obs[disease_key].astype(str)
    adata.obs[region_key] = adata.obs[region_key].astype(str)

    adata.obs["split_label"] = adata.obs[disease_key] + "-" + adata.obs[region_key]

    # === 4. Ensure consistent categories ===
    categories = adata.obs[celltype_key].cat.categories.tolist()
    adata.obs[celltype_key] = pd.Categorical(adata.obs[celltype_key], categories=categories)


    sc.tl.pca(adata, n_comps=20)
    sc.pp.neighbors(adata, use_rep='X_pca', n_pcs=20)
    sc.tl.umap(adata)


    # === 5. Save combined AnnData ===
    adata.write(output_combined)
    reference = adata[~(adata.obs[region_key] == 'premotor cortex')].copy()
    query = adata[adata.obs[region_key] == 'premotor cortex'].copy()
    print(f"OBSM check ==> Reference:{reference.obsm.keys()} Query: {query.obsm.keys()}")
    query.var = adata.var.copy()
    reference.var = adata.var.copy()
    unique_cts = adata.obs[celltype_key].cat.categories.tolist()
    query.obs[celltype_key] = pd.Categorical(query.obs[celltype_key], categories=unique_cts)
    reference.obs[celltype_key] = pd.Categorical(reference.obs[celltype_key], categories=unique_cts)

    print(f"Query     : {query.n_obs} cells × {query.n_vars} genes")
    print(f"Reference : {reference.n_obs} cells × {reference.n_vars} genes\n")

    print(f"Writing reference → {reference_out}")
    reference.write_h5ad(reference_out)

    print(f"Writing query     → {query_out}")
    query.write_h5ad(query_out)
    return reference, query


def get_stats(reference, query, celltype_key, summary_out):
    combined = query.obs.copy()
    combined['source'] = 'query'
    reference_obs = reference.obs.copy()
    reference_obs['source'] = 'reference'
    combined = pd.concat([combined, reference_obs], axis=0)

    # Group by split_label and celltype
    summary_df = (
        combined.groupby(['split_label', celltype_key])
        .size()
        .reset_index(name='count')
        .pivot(index='split_label', columns=celltype_key, values='count')
        .fillna(0)
        .astype(int)
    )

    # Add row sums
    summary_df['Total'] = summary_df.sum(axis=1)

    # Add column totals (including Total column)
    summary_df.loc['Total'] = summary_df.sum(axis=0)

    # Save as Excel (XLS)
    summary_df.to_excel(summary_out)

if __name__ == '__main__':
    root_dir = "scgpt/benchmark/ms"
    celltype_key = "Factor Value[inferred cell type - authors labels]"
    reference_out = f"{root_dir}/reference_annot.h5ad"
    query_out = f"{root_dir}/query_annot.h5ad"
    disease_key = "Sample Characteristic[disease]"
    region_key = "Sample Characteristic[sampling site]"
    if os.path.exists(reference_out) and os.path.exists(query_out):
        reference = sc.read_h5ad(reference_out)
        query = sc.read_h5ad(query_out)
    else:
        reference, query = prep(reference_path=f"{root_dir}/reference.h5ad",
             query_path=f"{root_dir}/query.h5ad",
                                output_combined = f"{root_dir}/ms_annot.h5ad")
    get_stats(reference, query, celltype_key, summary_out=f"{root_dir}stats.xlsx")