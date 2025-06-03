#!/bin/bash

agg_method="${1-fedavg}"
weighted="${2-true}"
smpc="${3-true}"
GPU=${4-0}

if [[ "$agg_method" != "fedavg" && "$agg_method" != "fedprox" ]]; then
    echo "Invalid aggregation method. Use 'fedavg' or 'fedprox'."
    exit 1
fi

declare -A datasets
datasets["MS"]="ms|reference.h5ad|query.h5ad|Factor Value[inferred cell type - authors labels]|Factor Value[sampling site]"
datasets["HP"]="hp|reference_refined.h5ad|query.h5ad|Celltype|batch"
datasets["MYELOID-top4+rest"]="myeloid|reference_adata.h5ad|query_adata.h5ad|combined_celltypes|top4+rest"

root_dir="$(dirname "$PWD")"
INTI_WEIGHTS_DIR="${root_dir}/models/init"
N_ROUNDS=20

smpc_subdir=""
if [ "$smpc" == "true" ]; then
    smpc_subdir="/smpc"
fi

param_tuning_res="${root_dir}/output/annotation/param-tuning-res"



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

    # 3) Build the base Python command (everything up through --param_tuning)
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

for key in "${!datasets[@]}"; do
    # Split the pipe‐separated fields into args[0..4]
    IFS='|' read -r -a args <<< "${datasets[$key]}"
    data_dir="${root_dir}/data/scgpt/benchmark/${args[0]}"
    reference="${data_dir}/${args[1]}"
    query="${data_dir}/${args[2]}"

    for epoch in 1 2 3 4 5; do
        if [ "$agg_method" == "fedavg" ]; then
            # FedAvg: no extra flags, subdir="param_tuning", description includes epoch
            desc="FEDAVG, epoch=${epoch}"
            run_job "$desc" "param_tuning" ""
            # Note: if run_job fails, it already printed an error. We can choose to continue/end.
            # if [ $? -ne 0 ]; then continue; fi

        else
            # FedProx: loop over all mu values
            for mu in 0.001 0.01 0.05 0.1 0.2 0.5; do
                desc="FEDPROX, epoch=${epoch}, mu=${mu}"
                run_job "$desc" "mu_tuning" "--mu $mu --use_fedprox"
                # if [ $? -ne 0 ]; then continue; fi
            done
        fi
    done
done
