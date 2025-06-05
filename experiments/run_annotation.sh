#!/bin/bash

# Assign command-line arguments to variables
mode="$1"
n_epochs="$2"
n_rounds="$3"
smpc="${4-false}"
GPU="${5-0}"

chmod +x annotation.sh

declare -A datasets
datasets["MS"]="ms|reference.h5ad|query.h5ad|Factor Value[inferred cell type - authors labels]|Factor Value[sampling site]"
datasets["HP"]="hp|reference_refined.h5ad|query.h5ad|Celltype|batch"
datasets["MYELOID-top4+rest"]="myeloid|reference_adata.h5ad|query_adata.h5ad|combined_celltypes|top4+rest"
datasets["COVID"]="covid|reference_annot.h5ad|query_annot.h5ad|celltype|str_batch"
datasets["LUNG"]="lung|reference_annot.h5ad|query_annot.h5ad|cell_type|sample"
datasets["CellLine"]="cl|reference.h5ad|query.h5ad|cell_type|batch"


echo "Running annotation for ${mode}"


for key in "${!datasets[@]}"; do
    echo -e "\e[32m******************************************\e[0m"
    echo -e "\e[32mRunning annotation for $key dataset [SMPC is ${smpc}]\e[0m"
    IFS='|' read -r -a args <<< "${datasets[$key]}"
    echo -e "\e[32mArguments: ${args[0]} ${args[1]} ${args[2]} ${args[3]} ${args[4]} ${args[5]}\e[0m"
    echo -e "\e[32m******************************************\e[0m"
    ./annotation.sh "${mode}" "${args[0]}" "${args[1]}" "${args[2]}" "${args[3]}" "${args[4]}" "${GPU}" "${n_epochs}" "${n_rounds}" "${smpc}"

    if [ $? -ne 0 ]; then
        echo -e "\e[31Error processing dataset $key. Please check the configuration.\e[0m"
        continue
    fi
done
