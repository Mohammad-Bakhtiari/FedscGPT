#!/bin/bash

# Assign command-line arguments to variables
datasetnames="${1}"
mode="$2"
n_epochs="$3"
n_rounds="$4"
smpc="${5-false}"
GPU="${6-0}"
agg_method="${7-fedavg}"
weighted="${8-false}"
mu="${9-0}"

chmod +x annotation.sh

declare -A datasets
datasets["MS"]="ms|reference_annot.h5ad|query_annot.h5ad|Factor Value[inferred cell type - authors labels]|split_label"
datasets["HP"]="hp|reference_refined.h5ad|query.h5ad|Celltype|batch"
datasets["MYELOID-top4+rest"]="myeloid|reference_adata.h5ad|query_adata.h5ad|combined_celltypes|top4+rest"
datasets["LUNG"]="lung|reference_annot.h5ad|query_annot.h5ad|cell_type|sample"
datasets["CellLine"]="cl|reference.h5ad|query.h5ad|cell_type|batch"
datasets["COVID"]="covid|reference_annot.h5ad|query_annot.h5ad|celltype|batch_group"
datasets["COVID-cent_corrected"]="covid-corrected|reference.h5ad|query.h5ad|celltype|batch_group"
datasets["COVID-fed-corrected"]="covid-fed-corrected|reference.h5ad|query.h5ad|celltype|batch_group"

IFS=',' read -ra keys <<< "$datasetnames"
if [[ "${datasetnames}" != "all" ]]; then
    for key in "${keys[@]}"; do
        if [[ -z "${datasets[$key]}" ]]; then
            echo "Dataset \"$key\" not found. Available keys: ${!datasets[@]}"
            exit 1
        fi
    done
else
    keys=("${!datasets[@]}")
fi


echo "Running annotation for ${mode}"


for key in "${keys[@]}"; do
    echo -e "\e[32m******************************************\e[0m"
    echo -e "\e[32mRunning annotation for $key dataset [SMPC is ${smpc}]\e[0m"
    IFS='|' read -r -a args <<< "${datasets[$key]}"
    echo -e "\e[32mArguments: ${args[0]} ${args[1]} ${args[2]} ${args[3]} ${args[4]} ${args[5]}\e[0m"
    echo -e "\e[32m******************************************\e[0m"
    ./annotation.sh "${mode}" "${args[0]}" "${args[1]}" "${args[2]}" "${args[3]}" "${args[4]}" "${GPU}" "${n_epochs}" "${n_rounds}" "${agg_method}" "${weighted}" "${smpc}" "${mu}"

    if [ $? -ne 0 ]; then
        echo -e "\e[31Error processing dataset $key. Please check the configuration.\e[0m"
        continue
    fi
done
