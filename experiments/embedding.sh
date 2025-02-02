#!/bin/bash

# Assign command-line arguments to variables
mode="$1"
dataset="$2"
reference_file="$3"
query_file="$4"
celltype_key="$5"
batch_key="$6"
gene_col="$7"
gpu="$8"

# Get the root directory, which is the parent directory of the current working directory
root_dir="$(dirname "$PWD")"

directory_mode="centralized"
if [[ "$mode" == *"federated"* ]]; then
  directory_mode="federated"
fi

# Set up directory paths
data_dir="${root_dir}/data/benchmark/${dataset}"
reference="${data_dir}/${reference_file}"
query="${data_dir}/${query_file}"
output="${root_dir}/output/embedding/${dataset}/${directory_mode}"

if [ ! -d "$output" ]; then
    mkdir -p "$output"
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
python ${root_dir}/tasks/embedding.py \
 --dataset_name $dataset \
 --data-dir $data_dir \
 --reference_adata $reference \
 --query_adata $query \
 --output-dir $output \
 --celltype_key "$celltype_key" \
 --batch_key "$batch_key" \
 --gene_col "$gene_col" \
 --mode $mode \
 --gpu $gpu \
 --pretrained_model_dir "${root_dir}/pretrained_models/scGPT_human" \
 --config_file "${root_dir}/experiments/configs/embedding/config.yml" \
 --fed_config_file "${root_dir}/experiments/configs/embedding/fed_config.yml"
