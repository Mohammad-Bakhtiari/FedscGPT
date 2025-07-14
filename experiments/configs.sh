#!/bin/bash

declare -A datasets
datasets["MS"]="ms|reference_annot.h5ad|query_annot.h5ad|Factor Value[inferred cell type - authors labels]|split_label|index"
datasets["HP"]="hp|reference_refined.h5ad|query.h5ad|Celltype|batch|index"
datasets["MYELOID-top4+rest"]="myeloid|reference_adata.h5ad|query_adata.h5ad|combined_celltypes|top4+rest|index"
datasets["LUNG"]="lung|reference_annot.h5ad|query_annot.h5ad|cell_type|sample|gene_name"
datasets["CellLine"]="cl|reference.h5ad|query.h5ad|cell_type|batch|index"
datasets["COVID"]="covid|reference-raw.h5ad|query-raw.h5ad|celltype|batch_group|gene_name"
datasets["COVID-corrected"]="covid-corrected|reference_corrected.h5ad|query_corrected.h5ad|celltype|batch_group|gene_name"
datasets["COVID-fed-corrected"]="covid-fed-corrected|reference_fed_corrected.h5ad|query_fed_corrected.h5ad|celltype|batch_group|gene_name"


# ----------------------
# Dataset selection logic
# Usage:
#   resolve_dataset_keys "dataset1,dataset2"
#   resolve_dataset_keys "all"
# Result:
#   Sets `keys` array variable with valid dataset names.
# ----------------------
resolve_dataset_keys() {
    local datasetnames="$1"
    IFS=',' read -ra input_keys <<< "$datasetnames"

    if [[ "$datasetnames" != "all" ]]; then
        for key in "${input_keys[@]}"; do
            if [[ -z "${datasets[$key]}" ]]; then
                echo "❌ Dataset \"$key\" not found."
                echo "✅ Available datasets: ${!datasets[@]}"
                exit 1
            fi
        done
        keys=("${input_keys[@]}")
        echo "✅ Selected datasets: ${keys[*]}"
    else
        keys=("${!datasets[@]}")
    fi
}
