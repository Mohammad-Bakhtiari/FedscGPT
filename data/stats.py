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
    },
    "hp": {
    },

}


def get_stats(df, celltype_key, batch_key, celltype_mapping, batch_map):
    # Safe partial mapping: use original value if not in map
    df["cell type"] = df[celltype_key].map(lambda x: celltype_mapping.get(x, x))
    df["batch"] = df[batch_key].map(lambda x: batch_map.get(x, x))
    summary_df = df.groupby(['cell type', 'batch']).size().unstack(fill_value=0)
    summary_df['Total'] = summary_df.sum(axis=1)
    summary_df.loc['Total'] = summary_df.sum(numeric_only=True)
    return summary_df

# Dataset configuration list
rootdir = "scgpt/benchmark"
datasets = {
    # "ms": {
    #     "h5ad_file": "ms_annot.h5ad",
    #     "celltype_key": "Factor Value[inferred cell type - authors labels]",
    #     "batch_key": "split_label",
    # },
    'covid-annotation': {
        "folder": "covid",
        "h5ad_file": "reference_annot.h5ad|query_annot.h5ad",
        "celltype_key": "celltype",
        "batch_key": "batch_group",
    },
    'covid-embedding': {
        "folder": "covid-emb",
        "h5ad_file": "reference.h5ad|query.h5ad",
        "celltype_key": "celltype",
        "batch_key": "str_batch",
    },
# datasets["HP"]="hp|reference_refined.h5ad|query.h5ad|Celltype|batch"
# datasets["MYELOID-top4+rest"]="myeloid|reference_adata.h5ad|query_adata.h5ad|combined_celltypes|top4+rest"
# datasets["LUNG"]="lung|reference_annot.h5ad|query_annot.h5ad|cell_type|sample"
# datasets["CellLine"]="cl|reference.h5ad|query.h5ad|cell_type|batch"
# datasets["COVID"]="covid|reference_annot.h5ad|query_annot.h5ad|celltype|batch_group"
# datasets["COVID-cent_corrected"]="covid-corrected|reference.h5ad|query.h5ad|celltype|batch_group"
# datasets["COVID-fed-corrected"]="covid-fed-corrected|reference.h5ad|query.h5ad|celltype|batch_group"
#     "hp": {
#
#         "h5ad_file": "reference_refined.h5ad|query.h5ad",
#         "celltype_key": "cell_type",
#         "batch_key": "batch_group",
#     },
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
    if len(files) == 1:
        file_path = os.path.join(ds_path, files[0])
        return sc.read_h5ad(file_path)

    elif len(files) == 2:
        reference_file, query_file = files
        reference_path = os.path.join(ds_path, reference_file)
        query_path = os.path.join(ds_path, query_file)

        reference = sc.read_h5ad(reference_path)
        query = sc.read_h5ad(query_path)

        return reference.concatenate(query, batch_key="query_ref_split_label", batch_categories=["reference", "query"])

    else:
        raise ValueError("files must contain one or two filenames")


with pd.ExcelWriter(output_excel_path) as writer:
    for dataset in datasets.keys():
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

