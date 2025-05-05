#!/bin/bash

# Usage: ./run_annotation.sh <DATASET> <N_EPOCHS> <N_ROUNDS> <GPU>
# Example: ./run_annotation.sh MS 5 20 0

# Input arguments
DATASET=$1
N_EPOCHS=$2
N_ROUNDS=$3
GPU=$4

# === Dataset configurations ===
declare -A datasets
datasets["MS"]="ms|reference.h5ad|query.h5ad|Factor Value[inferred cell type - authors labels]|Factor Value[sampling site]"
datasets["HP"]="hp|reference_refined.h5ad|query.h5ad|Celltype|batch"
datasets["MYELOID-top4+rest"]="myeloid|reference_adata.h5ad|query_adata.h5ad|combined_celltypes|top4+rest"

# Check if dataset exists
if [[ -z "${datasets[$DATASET]}" ]]; then
  echo "Error: Unknown dataset '$DATASET'"
  echo "Available datasets: ${!datasets[@]}"
  exit 1
fi

# === Set paths ===
root_dir="$(dirname "$PWD")"
INIT_WEIGHTS_DIR="${root_dir}/init_weights"

IFS='|' read -r -a args <<< "${datasets[$DATASET]}"

data_dir="${root_dir}/data/scgpt/benchmark/${args[0]}"
reference="${data_dir}/${args[1]}"
query="${data_dir}/${args[2]}"
output="${root_dir}/output/annotation/${args[0]}/param_tuning"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# === Run the Python script ===
python "${root_dir}/tasks/annotation.py" \
 --dataset_name "${args[0]}" \
 --data-dir "$data_dir" \
 --reference_adata "$reference" \
 --query_adata "$query" \
 --output-dir "$output" \
 --celltype_key "${args[3]}" \
 --batch_key "${args[4]}" \
 --mode "federated_finetune" \
 --gpu "$GPU" \
 --pretrained_model_dir "${root_dir}/models/pretrained_models/scGPT_human" \
 --config_file "${root_dir}/experiments/configs/annotation/config.yml" \
 --fed_config_file "${root_dir}/experiments/configs/annotation/fed_config.yml" \
 --n_epochs "$N_EPOCHS" \
 --n_rounds "$N_ROUNDS" \
 --param_tuning_res "${root_dir}/output/annotation/param-tuning-res" \
 --init_weights_dir "$INIT_WEIGHTS_DIR/${args[0]}.pth" \
 --param_tuning
