#!/bin/bash

# Assign command-line arguments to variables
mode="$1"
n_clients="$2"
gpu="$3"
n_epochs="${4}"
n_rounds="${5}"
per_round_eval="${6}"


perts_to_plot=""
# Get the root directory, which is the parent directory of the current working directory
root_dir="$(dirname "$PWD")"

directory_mode="centralized"
if [[ "$mode" == *"federated"* ]]; then
  directory_mode="federated"
fi

# Set up directory paths
data_dir="${root_dir}/data/benchmark/perturbation/norman"
reference="${data_dir}/reverse/perturb_processed.h5ad"
query="${data_dir}/perturb_processed.h5ad"
output="${root_dir}/output/perturbation/norman/${directory_mode}"
INTI_WEIGHTS_DIR="${root_dir}/init_weights"

if [ ! -d "$output" ]; then
    mkdir -p $output
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
cmd="python ${root_dir}/tasks/perturbation.py \
 --dataset_name reverse-norman \
 --data-dir $data_dir \
 --reference_adata $reference \
 --query_adata $query \
 --perts_to_plot $perts_to_plot \
 --pyg_path "${data_dir}/reverse/data_pyg" \
 --split_path "${data_dir}/reverse/splits" \
 --output-dir $output \
 --mode $mode \
 --gpu $gpu \
 --pretrained_model_dir ${root_dir}/pretrained_models/scGPT_human \
 --config_file ${root_dir}/experiments/configs/perturbation/config.yml \
 --n_clients $n_clients \
 --fed_config_file ${root_dir}/experiments/configs/perturbation/fed_config.yml \
 --init_weights_dir ${INTI_WEIGHTS_DIR}/${dataset}.pth \
 --reverse \
 --verbose"


# Add optional arguments if they are set
if [ "${#n_epochs}" != 0 ]; then
    cmd="$cmd --n_epochs $n_epochs"
fi

# Check and add --n_rounds if n_rounds is not empty and not "None"
if [ "${#n_rounds}" != 0 ]; then
    cmd="$cmd --n_rounds $n_rounds"
fi

if [ "${per_round_eval}" == True ]; then
    cmd="$cmd --per_round_eval"
fi
# Execute the command
eval $cmd
