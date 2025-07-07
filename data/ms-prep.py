import scanpy as sc
import pandas as pd
import numpy as np

# === Config ===
root_dir = "../ms"
query_path = f"{root_dir}/query.h5ad"
reference_path = f"{root_dir}/reference.h5ad"
celltype_key = "Factor Value[inferred cell type - authors labels]"
batch_key = "Factor Value[sampling site]"
output_combined = f"{root_dir}/ms.h5ad"

target_batch = "multiple sclerosis | premotor cortex"
disease_key = "Sample Characteristic[disease]"
region_key = "Sample Characteristic[sampling site]"


query = sc.read_h5ad(query_path)
reference = sc.read_h5ad(reference_path)

# === 2. Combine datasets ===
adata = reference.concatenate(query, batch_key=split_key, batch_categories=["reference", "query"])
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

# === 6. Save split-specific cell type counts ===
counts = (
    adata.obs
    .groupby([batch_key, split_key])[celltype_key]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)
counts.to_csv(output_celltype_stats, index=False)

# === 7. Save total cell type counts per batch ===
combined_counts = (
    adata.obs
    .groupby(batch_key)[celltype_key]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)
combined_counts.to_csv(output_combined_stats, index=False)

print("✅ Done. Files written:")
print(f" - {output_combined}")
print(f" - {output_celltype_stats}")
print(f" - {output_combined_stats}")
