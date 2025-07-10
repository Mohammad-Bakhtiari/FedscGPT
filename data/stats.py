import os
import scanpy as sc
import pandas as pd

# Mapping for cell types
celltype_mapping = {
    ms: {
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

}


def get_stats(df, celltype_key, batch_key, celltype_mapping, batch_map):
    df["cell type"] = df[celltype_key].map(celltype_mapping)
    df["batch"] = df[batch_key].map(batch_map)
    summary_df = adata.groupby(['cell type', 'batch']).size().unstack(fill_value=0)
    summary_df['Total'] = summary_df.sum(axis=1)
    summary_df.loc['Total'] = summary_df.sum(numeric_only=True)
    return summary_df

# Dataset configuration list
rootdir = "scgpt/benchmark"
datasets = {
    "ms": {
        "h5ad_file": "ms_annot.h5ad",
        "celltype_key": "Factor Value[inferred cell type - authors labels]",
        "batch_key": "split_label",
    },
}

# Output Excel file with multiple sheets
output_excel_path = "summary_stats.xlsx"
with pd.ExcelWriter(output_excel_path) as writer:
    for dataset in datasets.keys():
        adata = sc.read_h5ad(os.path.join(rootdir, datasets[dataset]["h5ad_file"]))
        stats_df = get_stats(adata.obs,
                             celltype_key=datasets[dataset]["celltype_key"],
                             batch_key=datasets[dataset]["batch_key"],
                             celltype_mapping=celltype_mapping[dataset],
                             batch_map=batch_map[dataset]
                             )
        stats_df.to_excel(writer, sheet_name=dataset)

