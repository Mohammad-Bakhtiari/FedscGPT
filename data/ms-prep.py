import scanpy as sc
import pandas as pd
import numpy as np
from prep_batch_effect_correction import ref_query_split

# === Config ===
root_dir = "scgpt/benchmark/ms"
query_path = f"{root_dir}/query.h5ad"
reference_path = f"{root_dir}/reference.h5ad"
celltype_key = "Factor Value[inferred cell type - authors labels]"
batch_key = "Factor Value[sampling site]"
output_combined = f"{root_dir}/ms_annot.h5ad"
reference_out = f"{root_dir}/reference_annot.h5ad"
query_out = f"{root_dir}/query_annot.h5ad"

disease_key = "Sample Characteristic[disease]"
region_key = "Sample Characteristic[sampling site]"


query = sc.read_h5ad(query_path)
reference = sc.read_h5ad(reference_path)

# === 2. Combine datasets ===
adata = reference.concatenate(query, batch_key='split_label', batch_categories=["reference", "query"])
print(f"Combined shape: {adata.shape}")

adata.obs[disease_key] = adata.obs[disease_key].astype(str)
adata.obs[region_key] = adata.obs[region_key].astype(str)

adata.obs["split_label"] = adata.obs[disease_key] + "-" + adata.obs[region_key]

split_counts = (adata.obs.groupby(["split_label", celltype_key]).size().unstack(fill_value=0))
max_split_per_celltype = split_counts.idxmax().reset_index(name="max_split_label")
max_count_per_celltype = split_counts.max().reset_index(name="max_count")
summary = pd.merge(max_split_per_celltype, max_count_per_celltype, on=celltype_key)

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

# === 6. Save split-specific cell type counts ===
counts = (
    adata.obs
    .groupby([batch_key, 'split_label'])[celltype_key]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)
counts.to_csv(f"{root_dir}/counts.csv", index=False)

# === 7. Save total cell type counts per batch ===
combined_counts = (
    adata.obs
    .groupby(batch_key)[celltype_key]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)
combined_counts.to_csv(f"{root_dir}/combined_counts.csv", index=False)

print("✅ Done. Files written:")
print(f" - {output_combined}")
print(f" - {output_celltype_stats}")
print(f" - {output_combined_stats}")
