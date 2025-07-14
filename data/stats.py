import os
import scanpy as sc
import pandas as pd

# Mapping for cell types
celltype_mapping = {
    'ms': {
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
    },
    "covid": {
    },
    "hp": {
    },

}

# Placeholder for batch mapping (to be customized)
batch_map = {
    "ms":{
        "Ctrl_Premotor": "Control Premotor",
        "MS_Premotor": "MS Premotor",
        "Ctrl_Prefrontal": "Control Prefrontal",
        "MS_Prefrontal": "MS Prefrontal",
        "Ctrl_Cerebral": "Control Cerebral",
        "MS_Cerebral": "MS Cerebral",
    },
    "covid": {
        'Sanger_Meyer_2019Madissoon': 'Sanger',
        'COVID-19 (query)': 'COVID',
        'Northwestern_Misharin_2018Reyfman': 'Northwestern',
    },
    "hp": {
        '0': 'Baron',
        '1': 'Mutaro',
        '2': 'Segerstolpe',
        '3': 'Wang',
        '4': 'Xin'
    },

}
batch_map['covid-corrected'] = batch_map['covid']
batch_map['covid-fed-corrected'] = batch_map['covid']

def get_stats(df, celltype_key, batch_key, celltype_mapping, batch_map):
    """
    Generates a summary of cell type counts across Reference and Query batches.

    Parameters:
    - df: pandas DataFrame (e.g., `adata.obs`)
    - celltype_key: column name for cell type annotations
    - batch_key: column name for sample IDs or batch identifiers (e.g., 'sample')
    - celltype_mapping: dictionary to rename cell types (optional)
    - batch_map: dictionary to rename batch keys (optional)

    Returns:
    - summary_df: pandas DataFrame with hierarchical columns (Reference/Query → Sample IDs),
                  with Query batches ordered after Reference batches.
    """
    # Step 1: Rename cell types and batch names using mappings
    df = df.copy()
    df["cell type"] = df[celltype_key].map(lambda x: celltype_mapping.get(x, x))
    df["batch"] = df[batch_key].map(lambda x: batch_map.get(x, x))

    # Step 2: Group by cell type and batch, then pivot to wide format
    summary = df.groupby(["cell type", "batch"]).size().unstack(fill_value=0)

    # Step 3: Map each batch to 'Reference' or 'Query'
    batch_to_split = df.drop_duplicates(subset="batch").set_index("batch")["query_ref_split_label"].to_dict()

    # Step 4: Assign hierarchical columns and sort with Query batches last
    new_cols = [(batch_to_split.get(col, "Unknown"), col) for col in summary.columns]
    sorted_cols = sorted(new_cols, key=lambda x: (0 if x[0] == "Reference" else 1, x[1]))
    summary.columns = pd.MultiIndex.from_tuples(sorted_cols)

    # Step 5: Add total column per cell type
    summary[("", "Total")] = summary.sum(axis=1)

    # Step 6: Add total row at bottom
    summary.loc["Total"] = summary.sum(numeric_only=True)

    return summary



# Dataset configuration list
rootdir = "scgpt/benchmark"
datasets = {
    "ms": {
        "h5ad_file": "reference_annot.h5ad|query_annot.h5ad",
        "celltype_key": "Factor Value[inferred cell type - authors labels]",
        "batch_key": "split_label",
    },
    'covid': {
        "folder": "covid",
        "h5ad_file": "reference-raw.h5ad|query-raw.h5ad",
        "celltype_key": "celltype",
        "batch_key": "batch_group",
    },

    "hp": {

        "h5ad_file": "reference_refined.h5ad|query.h5ad",
        "celltype_key": "Celltype",
        "batch_key": "batch",
    },
    "lung": {
        "h5ad_file": "reference_annot.h5ad|query_annot.h5ad",
        "celltype_key": "cell_type",
        "batch_key": "sample",
    },
    "myeloid": {
        "h5ad_file": "reference_adata.h5ad|query_adata.h5ad",
        "celltype_key": "combined_celltypes",
        "batch_key": "top4+rest",
    },
    "cl": {
        "h5ad_file": "reference.h5ad|query.h5ad",
        "celltype_key": "cell_type",
        "batch_key": "batch",
    },
    "covid-corrected": {
        "h5ad_file": "reference_corrected.h5ad|query_corrected.h5ad",
        "celltype_key": "celltype",
        "batch_key": "batch_group",
    },
    "covid-fed-corrected": {
        "h5ad_file": "reference_fed_corrected.h5ad|query_fed_corrected.h5ad",
        "celltype_key": "celltype",
        "batch_key": "batch_group",
    },
}

# Output Excel file with multiple sheets
output_excel_path = "summary_stats.xlsx"



def read_adata(files, ds_path):
    """
    Reads and concatenates h5ad files for a given dataset.

    Parameters:
    - files: list of filenames (1 or 2 .h5ad files)
    - ds_path: path to the dataset directory

    Returns:
    - AnnData object (single or concatenated)
    """
    if len(files) == 2:
        reference_file, query_file = files
        reference_path = os.path.join(ds_path, reference_file)
        query_path = os.path.join(ds_path, query_file)

        reference = sc.read_h5ad(reference_path)
        query = sc.read_h5ad(query_path)

        return reference.concatenate(query, batch_key="query_ref_split_label", batch_categories=["Reference", "Query"])

    else:
        raise ValueError("files must contain one or two filenames")


if __name__ == '__main__':
    with pd.ExcelWriter(output_excel_path) as writer:
        for dataset in datasets.keys():
            folder = datasets[dataset].get("folder", dataset)
            ctm = celltype_mapping.get(folder, {})
            bm = batch_map.get(folder, {})
            print(folder, dataset)
            if dataset == "myeloid":
                ref_file, q_file = datasets[dataset]["h5ad_file"].split("|")
                ref = sc.read_h5ad(os.path.join(rootdir, folder, ref_file))
                ref.obs["query_ref_split_label"] = "Reference"
                stats_df = get_stats(ref.obs,
                                     celltype_key=datasets[dataset]["celltype_key"],
                                     batch_key=datasets[dataset]["batch_key"],
                                     celltype_mapping=ctm,
                                     batch_map=bm
                                     )
                stats_df.to_excel(writer, sheet_name=dataset + " Reference")
                print(f"######### Statistics for {dataset}: Reference #########")
                print(stats_df)
                print("#" * 50)
                query = sc.read_h5ad(os.path.join(rootdir, folder, q_file))
                query.obs["query_ref_split_label"] = "Query"
                stats_df_query = get_stats(query.obs,
                                           celltype_key=datasets[dataset]["celltype_key"],
                                           batch_key=datasets[dataset]["batch_key"],
                                           celltype_mapping=ctm,
                                           batch_map=bm
                                           )
                stats_df_query.to_excel(writer, sheet_name=dataset + " Query")
                print(f"######### Statistics for {dataset}: Query #########")
                print(stats_df_query)
                print("#" * 50)
            else:
                adata = read_adata(datasets[dataset]["h5ad_file"].split("|"), os.path.join(rootdir, folder))
                stats_df = get_stats(adata.obs,
                                     celltype_key=datasets[dataset]["celltype_key"],
                                     batch_key=datasets[dataset]["batch_key"],
                                     celltype_mapping=ctm,
                                     batch_map=bm
                                     )
                stats_df.to_excel(writer, sheet_name=dataset)
                print(f"######### Statistics for {dataset}: #########")
                print(stats_df)
                print("#" * 50)

