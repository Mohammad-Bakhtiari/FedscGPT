#!/bin/bash

data_dir="$1"

declare -A datasets
#datasets["CellLine"]="cl|CellLine.h5ad|reference.h5ad|query.h5ad|cell_type|batch|2|true|min_max"
#datasets["COVID"]="covid|Covid_annot.h5ad|reference_annot.h5ad|query_annot.h5ad|celltype|ref-query-split|q|true|min_max"
datasets["LUNG"]="lung|Lung_annot.h5ad|reference_annot.h5ad|query_annot.h5ad|cell_type|ref-query-split|q|true|min_max"

for key in "${!datasets[@]}"; do
    echo -e "\e[32m******************************************\e[0m"
    echo -e "\e[32mRunning preprocessing for $key dataset\e[0m"
    echo -e "\e[32m******************************************\e[0m"

    # Parse the configuration string into an array
    IFS='|' read -r -a args <<< "${datasets[$key]}"

    subdir="${args[0]}"
    orig_file="${args[1]}"
    reference_filename="${args[2]}"
    query_filename="${args[3]}"
    celltype_key="${args[4]}"
    batch_key="${args[5]}"
    query_batch="${args[6]}"
    normalize="${args[7]}"
    norm_method="${args[8]}"
    orig_path="$data_dir/$subdir/$orig_file"
    output_dir="$data_dir/$subdir"

    # Build the command to run prep.py
    cmd="python prep.py \
      --orig_path \"$orig_path\" \
      --reference_file \"$reference_filename\" \
      --query_file \"$query_filename\" \
      --output_dir \"$output_dir\" \
      --celltype_key \"$celltype_key\" \
      --batch_key \"$batch_key\" \
      --query_batch $query_batch"

    if [ "$normalize" = "true" ]; then
      cmd+=" --normalize --norm_method $norm_method"
    fi

    echo "Command: $cmd"
    eval $cmd
done
