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
    split_label_map = {
        "normal-premotor cortex": "Ctrl_Premotor",
        "multiple sclerosis-premotor cortex": "MS_Premotor",
        "normal-prefrontal cortex": "Ctrl_Prefrontal",
        "multiple sclerosis-prefrontal cortex": "MS_Prefrontal",
        "normal-cerebral cortex": "Ctrl_Cerebral",
        "multiple sclerosis-cerebral cortex": "MS_Cerebral",
    }
    # ensure that all split labels are mapped correctly
    assert set(adata.obs['split_label'].unique()) == set(split_label_map.keys()), "Mismatch in split labels."
    adata.obs['split_label'] = adata.obs['split_label'].map(split_label_map)


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

celltype_mapping = {
    'PVALB-expressing interneuron': 'PVALB interneuron',
    'SST-expressing interneuron': 'SST interneuron',
    'VIP-expressing interneuron': 'VIP interneuron',
    'SV2C-expressing interneuron': 'SV2C interneuron',
    'cortical layer 2-3 excitatory neuron A': 'L2/3 excitatory A',
    'cortical layer 2-3 excitatory neuron B': 'L2/3 excitatory B',
    'cortical layer 4 excitatory neuron': 'L4 excitatory',
    'cortical layer 5-6 excitatory neuron': 'L5/6 excitatory',
    'pyramidal neuron?': 'Pyramidal neuron',
    'mixed excitatory neuron': 'Mixed excitatory',
    'oligodendrocyte A': 'Oligodendrocyte A',
    'oligodendrocyte C': 'Oligodendrocyte C',
    'oligodendrocyte precursor cell': 'OPC',
    'astrocyte': 'Astrocyte',
    'microglial cell': 'Microglia',
    'phagocyte': 'Phagocyte',
    'endothelial cell': 'Endothelial',
    'mixed glial cell?': 'Mixed glia'
}


def get_stats(reference, query, celltype_key, summary_out):
    combined = query.obs.copy()
    combined['source'] = 'query'
    reference_obs = reference.obs.copy()
    reference_obs['source'] = 'reference'
    combined = pd.concat([combined, reference_obs], axis=0)
    combined["cell type"] = combined[celltype_key].map(celltype_mapping)
    combined.drop(columns=[celltype_key], inplace=True)

    summary_df = combined.groupby(['cell type', 'split_label']).size().unstack(fill_value=0)
    # Add row and column totals
    summary_df['Total'] = summary_df.sum(axis=1)
    summary_df.loc['Total'] = summary_df.sum(numeric_only=True)

    # Save as Excel (XLS)
    summary_df.to_excel(summary_out)
    print(summary_df)




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
    get_stats(reference, query, celltype_key, summary_out=f"{root_dir}/stats.xlsx")