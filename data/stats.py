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
        '1': 'Muraro',
        '2': 'Segerstolpe',
        '3': 'Wang',
        '4': 'Xin'
    },
    "hp5": {
        'Baron': 'Baron',
        'Mutaro': 'Muraro',
        'Segerstolpe': 'Segerstolpe',
        'Wang': 'Wang',
        'Xin': 'Xin'
    }

}
batch_map['covid-corrected'] = batch_map['covid']
batch_map['covid-fed-corrected'] = batch_map['covid']

def get_stats(df, celltype_key, batch_key, celltype_mapping=None, batch_map=None):
    """
    Build a summary table of cell type counts across batches grouped by Reference/Query,
    using explicit construction to avoid mapping issues.
    """
    df = df.copy()

    # Step 1: Apply mappings
    df["cell type"] = df[celltype_key].map(lambda x: celltype_mapping.get(x, x)) if celltype_mapping else df[celltype_key]
    df["batch"] = df[batch_key].map(lambda x: batch_map.get(x, x)) if batch_map else df[batch_key]

    # Step 2: Extract original batch and split info
    df["original_batch"] = df[batch_key]
    if "query_ref_split_label" not in df.columns:
        raise ValueError("Column 'query_ref_split_label' is missing.")
    df["split"] = df["original_batch"].map(df.drop_duplicates(subset="original_batch").set_index("original_batch")["query_ref_split_label"])

    # Step 3: Get unique sorted batches by split type
    batch_order = df.drop_duplicates(["batch", "split"])[["batch", "split"]]
    batch_order = batch_order.sort_values(by=["split", "batch"])
    batch_tuples = [(row["split"], row["batch"]) for _, row in batch_order.iterrows()]

    # Step 4: Pivot counts manually
    counts = df.groupby(["cell type", "batch"]).size().unstack(fill_value=0)

    # Step 5: Create final DataFrame with hierarchical columns
    all_celltypes = sorted(counts.index.tolist())
    summary = pd.DataFrame(index=all_celltypes)

    for split, batch in batch_tuples:
        if batch in counts.columns:
            summary[(split, batch)] = counts[batch]
        else:
            summary[(split, batch)] = 0

    # Step 6: Add total column and row
    summary[("", "Total")] = summary.sum(axis=1)
    summary.loc["Total"] = summary.select_dtypes(include="number").sum()

    # Step 7: Apply column hierarchy
    summary.columns = pd.MultiIndex.from_tuples(summary.columns)

    return summary





# Dataset configuration list
rootdir = "/home/mohammad/PycharmProjects/FedscGPT/data/"
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
    "hp5": {
        "h5ad_file": "reference.h5ad|query.h5ad",
        "celltype_key": "Celltype",
        "batch_key": "batch_name",
    },
    "lung": {
        "h5ad_file": "reference_annot.h5ad|query_annot.h5ad",
        "celltype_key": "cell_type",
        "batch_key": "sample",
    },
    "cl": {
        "h5ad_file": "reference.h5ad|query.h5ad",
        "celltype_key": "cell_type",
        "batch_key": "batch",
    },
    "myeloid-top5": {
        "h5ad_file": "reference.h5ad|query.h5ad",
        "celltype_key": "cell_type",
        "batch_key": "combined_batch",
    },
    "myeloid-top10": {
        "h5ad_file": "reference.h5ad|query.h5ad",
        "celltype_key": "cell_type",
        "batch_key": "combined_batch",
    },
    "myeloid-top20": {
        "h5ad_file": "reference.h5ad|query.h5ad",
        "celltype_key": "cell_type",
        "batch_key": "combined_batch",
    },
    "myeloid-top30": {
        "h5ad_file": "reference.h5ad|query.h5ad",
        "celltype_key": "cell_type",
        "batch_key": "combined_batch",
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
        for dataset in ['myeloid-top5']:#datasets.keys():
            folder = datasets[dataset].get("folder", dataset)
            ctm = celltype_mapping.get(folder, {})
            bm = batch_map.get(folder, {})
            print(folder, dataset)
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

