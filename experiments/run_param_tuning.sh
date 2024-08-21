#!/bin/bash


# Declare an associative array to store dataset configurations
declare -A datasets

# Add dataset configurations
# format: data_folder|adata_file|test_adata_file|celltype_key|batch_key|gpu
datasets["MS"]="ms|reference.h5ad|query.h5ad|Factor Value[inferred cell type - authors labels]|Factor Value[sampling site]"
datasets["HP"]="hp|reference_refined.h5ad|query.h5ad|Celltype|batch"
datasets["MYELOID-top4+rest"]="myeloid|reference_adata.h5ad|query_adata.h5ad|combined_celltypes|top4+rest"

root_dir="$(dirname "$PWD")"
INTI_WEIGHTS_DIR="${root_dir}/init_weights"
GPU=1
N_ROUNDS=20
# Loop through each dataset configuration
for key in "${!datasets[@]}"; do
    echo "Running data annotation for $key"

    for epoch in 1 2 3 4 5; do

        IFS='|' read -r -a args <<< "${datasets[$key]}"

        # Set up directory paths
        data_dir="${root_dir}/data/benchmark/${args[0]}"
        reference="${data_dir}/${args[1]}"
        query="${data_dir}/${args[2]}"
        output="${root_dir}/output/annotation/${args[0]}/param_tuning"

        export CUBLAS_WORKSPACE_CONFIG=:4096:8
        # Run the Python script with the specified arguments
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
         --pretrained_model_dir "${root_dir}/pretrained_models/scGPT_human" \
         --config_file "${root_dir}/experiments/configs/annotation/config.yml" \
         --fed_config_file "${root_dir}/experiments/configs/annotation/fed_config.yml" \
         --n_epochs "$epoch" \
         --n_rounds "$N_ROUNDS" \
         --param_tuning_res "${root_dir}/output/annotation/param-tuning-res" \
         --init_weights_dir "$INTI_WEIGHTS_DIR/${args[0]}.pth" \
         --param_tuning

        if [ $? -ne 0 ]; then
            echo "Error processing dataset $key. Please check the configuration."
            continue
        fi
    done
done

echo "Data annotation completed for all datasets."
