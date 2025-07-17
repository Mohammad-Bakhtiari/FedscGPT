#!/bin/bash

source ./configs.sh

datasetnames="${1-all}"
agg_method="${2-fedavg}"
weighted="${3-true}"
smpc="${4-true}"
N_ROUNDS="${5-20}"
GPU=${6-0}
epochs="${7-1-5}"


if [[ "$agg_method" != "fedavg" && "$agg_method" != "fedprox" ]]; then
    echo "Invalid aggregation method. Use 'fedavg' or 'fedprox'."
    exit 1
fi

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

epochs_values=()

# Function to expand a range like 3-5 to 3 4 5
expand_range() {
  local start=${1%-*}
  local end=${1#*-}
  for ((i=start; i<=end; i++)); do
    epochs_values+=("$i")
  done
}

# Split input by comma
IFS=',' read -ra parts <<< "$epochs"

for part in "${parts[@]}"; do
  if [[ "$part" =~ ^[0-9]+-[0-9]+$ ]]; then
    expand_range "$part"
  elif [[ "$part" =~ ^[0-9]+$ ]]; then
    epochs_values+=("$part")
  fi
done



root_dir="$(dirname "$PWD")"
INTI_WEIGHTS_DIR="${root_dir}/models/init"

smpc_subdir=""
if [ "$smpc" == "true" ]; then
    smpc_subdir="/smpc"
fi

param_tuning_res="${root_dir}/output/annotation/results_summary"



run_job() {
    local desc="$1"
    local subdir="$2"
    local extra_flags="$3"

    # 1) Banner
    echo -e "\e[32m******************************************\e[0m"
    echo -e "\e[32mRunning ${desc} on \"${key}\" dataset\e[0m"
    echo -e "\e[32m******************************************\e[0m"

    # 2) Ensure output directory exists
    output="${root_dir}/output/annotation/${args[0]}/${subdir}${smpc_subdir}"
    mkdir -p "$output"

    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    base_cmd="python \"${root_dir}/tasks/annotation.py\" \
  --dataset_name \"${args[0]}\" \
  --data-dir \"$data_dir\" \
  --reference_adata \"$reference\" \
  --query_adata \"$query\" \
  --output-dir \"$output\" \
  --celltype_key \"${args[3]}\" \
  --batch_key \"${args[4]}\" \
  --mode federated_finetune \
  --gpu $GPU \
  --pretrained_model_dir \"${root_dir}/models/pretrained_models/scGPT_human\" \
  --config_file \"${root_dir}/experiments/configs/annotation/config.yml\" \
  --fed_config_file \"${root_dir}/experiments/configs/annotation/fed_config.yml\" \
  --n_epochs $epoch \
  --n_rounds $N_ROUNDS \
  --param_tuning_res \"$param_tuning_res\" \
  --init_weights_dir \"$INTI_WEIGHTS_DIR/${args[0]}.pth\" \
  --param_tuning $extra_flags"

    # 4) Append "--smpc" if requested
    if [ "$smpc" == "true" ]; then
        base_cmd+=" --smpc"
    fi

    # 5) Append "--weighted" if requested
    if [ "$weighted" == "true" ]; then
        base_cmd+=" --weighted"
    fi

    # 6) Finally, run it
    eval "$base_cmd"
}

for key in "${keys[@]}"; do
    # Split the pipe‐separated fields into args[0..4]
    IFS='|' read -r -a args <<< "${datasets[$key]}"
    data_dir="${root_dir}/data/scgpt/benchmark/${args[0]}"
    reference="${data_dir}/${args[1]}"
    query="${data_dir}/${args[2]}"

    for epoch in "${epochs_values[@]}"; do
        if [ "$agg_method" == "fedavg" ]; then
            # FedAvg: no extra flags, subdir="param_tuning", description includes epoch
            desc="FEDAVG, epoch=${epoch}"
            run_job "$desc" "param_tuning" ""
        else
            # FedProx: loop over all mu values
            for mu in 0.01 0.05 0.1 0.2 0.5; do
                desc="FEDPROX, epoch=${epoch}, mu=${mu}"
                run_job "$desc" "mu_tuning" "--mu $mu --use_fedprox"
            done
        fi
    done
done
