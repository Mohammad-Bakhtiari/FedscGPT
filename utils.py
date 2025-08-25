import scanpy as sc
import numpy as np


def prep_ms_data():
    adata = sc.read("c_data.h5ad")
    adata_test = sc.read("filtered_ms_adata.h5ad")

    adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
    adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
    adata.var.set_index(adata.var["gene_name"], inplace=True)
    adata_test.var.set_index(adata.var["gene_name"], inplace=True)
    adata_combined = adata.concatenate(adata_test, batch_key="str_batch")

    umap_coords = np.full((adata_combined.shape[0], adata_test.obsm['X_umap'].shape[1]), np.nan)
    umap_coords[adata_combined.obs["batch_id"] == '1'] = adata_test.obsm['X_umap']
    adata_combined.obsm['X_umap'] = umap_coords

    pca_coords = np.full((adata_combined.shape[0], adata_test.obsm['X_pca'].shape[1]), np.nan)
    pca_coords[adata_combined.obs["batch_id"] == '1'] = adata_test.obsm['X_pca']
    adata_combined.obsm['X_pca'] = pca_coords

    # Save the concatenated AnnData object
    adata_combined.write("ms.h5ad")

    # Verify the integrity of the combined data
    print("Combined adata shape:", adata_combined.shape)
    print("UMAP coordinates shape:", adata_combined.obsm['X_umap'].shape)
    print("Unique batch IDs:", adata_combined.obs["batch_id"].unique())


def get_top_batches(adata, n, rest=False):
    batch_counts = adata.obs['batch'].value_counts()
    new_batch_labels = {str(i): b for i, b in enumerate(batch_counts.index)}
    top_n = new_batch_labels[:n]
    batch_obs_label = f"top{n}"
    if not rest:
        adata = adata[adata.obs.batch.isin(top_n.values())].copy()
        adata.obs[batch_obs_label] = adata.obs['batch'].map(new_batch_labels)
        return adata
    batch_obs_label = f"{batch_obs_label}+rest"
    adata.obs[batch_obs_label] = str(n + 1)
    adata.obs.at[adata.obs['batch'].isin(top_n.index), batch_obs_label] = adata.obs['batch'].map(new_batch_labels)
    return adata


def get_top_batches(adata, n, rest=False):
    # Get the count of each batch
    batch_counts = adata.obs['batch'].value_counts()
    new_batch_labels = {b: str(i) for i, b in enumerate(batch_counts.index.tolist())}
    top_n = {k: new_batch_labels[k] for k in list(new_batch_labels)[:n]}
    category = list(top_n.values())
    if rest:
        category.append(str(n))
    category = sorted(category)
    batch_obs_label = f"top{n}" + ('' if not rest else '+rest')
    if not rest:
        adata = adata[adata.obs['batch'].isin(top_n.keys())].copy()
    adata.obs[batch_obs_label] = adata.obs['batch'].map(new_batch_labels)
    if rest:
        adata.obs.loc[~adata.obs['batch'].isin(top_n.keys()), batch_obs_label] = str(n)
    adata.obs[batch_obs_label] = adata.obs[batch_obs_label].astype('category').cat.set_categories(category)
    return adata


def prep_myeloid():
    adata = sc.read("refernce_adata.h5ad")
    adata_test = sc.read("query_adata.h5ad")
    get_top_batches(adata, 5)
    # For a more granular grouping, you can also consider top 10 and top 20 as separate groups
    top_10 = batch_counts[:10].index.tolist()
    top_20 = batch_counts[:20].index.tolist()

    # Create new columns for these groups
    adata.obs['client_top_10'] = 'rest'
    adata.obs['client_top_20'] = 'rest'

    # Assign 'top_10' or 'top_20' to the respective batches
    adata.obs.loc[adata.obs['batch'].isin(top_10), 'client_top_10'] = adata.obs.loc[
        adata.obs['batch'].isin(top_10), 'batch'].astype(str)
    adata.obs.loc[adata.obs['batch'].isin(top_20), 'client_top_20'] = adata.obs.loc[
        adata.obs['batch'].isin(top_20), 'batch'].astype(str)

    # Verify the changes
    print(adata.obs[['batch', 'client', 'client_top_10', 'client_top_20']].head())


def prep_hp(adata, adata_test):
    celltype_mapppings = {'PP': 'PP', 'PSC': 'PSC', 'acinar': 'Acinar', 'alpha': 'Alpha', 'beta': 'Beta', 'delta': 'Delta',
                        'ductal': 'Ductal', 'endothelial': 'Endothelial', 'epsilon': 'Epsilon', 'mast': 'Mast',
                        'MHC class II': 'MHC class II', 't_cell': 'T_cell', 'schwann': 'Schwann', 'macrophage': 'Macrophage'}
    adata.obs['Celltype'] = adata.obs['Celltype'].map(celltype_mapppings)
    adata.obs['Celltype'] = adata.obs['Celltype'].astype('category')
    adata_test.obs['Celltype'] = adata_test.obs['Celltype'].map(celltype_mapppings)
    adata_test.obs['Celltype'] = adata_test.obs['Celltype'].astype('category')
    cell_types = ['PP', 'PSC', 'Acinar', 'Alpha', 'Beta', 'Delta', 'Ductal', 'Endothelial', 'Epsilon', 'Mast', 'MHC class II']
    adata = adata[adata.obs['Celltype'].isin(cell_types)].copy()
    adata.obs.Celltype = adata.obs.Celltype.cat.set_categories(cell_types)
    adata_test.obs.Celltype = adata_test.obs.Celltype.cat.set_categories(cell_types)
    adata.write("cliftiGPT/data/benchmark/hp/reference_refined.h5ad")
    adata_test.write("cliftiGPT/data/benchmark/hp/query.h5ad")
