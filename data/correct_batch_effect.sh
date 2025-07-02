#!/bin/bash
set -e
data_dir="$1"

declare -A datasets
# order: dataset_subdir|raw_filename|preprocessed_for_be_filename|celltype_key|batch_key|BATCHES
datasets["COVID"]="covid|Covid_annot.h5ad|preprocessed_for_be.h5ad|celltype|study|HCL,Krasnow,Sanger_Meyer_2019Madissoon,COVID-19 (query),Sun,10X,Oetjen,Northwestern_Misharin_2018Reyfman,Freytag"
datasets["MS"]="ms|refined_ms.h5ad|preprocessed_for_be.h5ad|Factor Value[inferred cell type - authors labels]|Factor Value[sampling site]|prefrontal cortex,cerebral cortex,premotor cortex"



for dataset in "${!datasets[@]}"; do
    echo -e "\e[32m******************************************\e[0m"
    echo -e "\e[32mRunning preprocessing for $dataset dataset\e[0m"
    echo -e "\e[32m******************************************\e[0m"

    IFS='|' read -r -a args <<< "${datasets[$dataset]}"

    ds_name="${args[0]}"
    ds_data_dir="${data_dir}/${ds_name}"
    query_file_prefix="${ds_data_dir}/query"
    reference_file_prefix="${ds_data_dir}/reference"
    prep_for_be_datapath="${ds_data_dir}/${args[2]}"
    celltype_key="${args[3]}"

    echo -e "\e[32mPreprocessing for $dataset completed.\e[0m"
    python prep_batch_effect_correction.py \
            --prep_type "pre" \
            --dataset "${ds_name}" \
            --raw_data_path "${ds_data_dir}/${args[1]}" \
            --prep_for_be_datapath "${prep_for_be_datapath}" \
            --celltype_key "${celltype_key}" \
            --batch_key "${args[4]}" \
            --reference_file "${reference_file_prefix}-raw.h5ad" \
            --query_file "${query_file_prefix}-raw.h5ad"

    echo -e "\e[32mBatch effect correction of $dataset.\e[0m"
    BATCHES="${args[5]}"
    n_clients=$(echo "$BATCHES" | tr ',' '\n' | wc -l)
    output="${ds_data_dir}/output"
    mkdir -p "$output"
    init_model_path="${output}/model"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8

    echo -e "\e[32mCentralized batch effect correction of $dataset.\e[0m"
    output_path="${output}/centralized"
    mkdir -p "$output_path"
    fedscgen-centralized \
      --model_path "$init_model_path" \
      --data_path "${prep_for_be_datapath}" \
      --output_path "$output_path" \
      --epoch 100 \
      --batch_key "batch_group" \
      --cell_key "${celltype_key}" \
      --z_dim 10 \
      --hidden_layers_sizes "800,800" \
      --batch_size 50 \
      --remove_cell_types "" \
      --early_stopping_kwargs "{'early_stopping_metric': 'val_loss', 'patience': 20, 'threshold': 0, 'reduce_lr': True, 'lr_patience': 13, 'lr_factor': 0.1}" \
      --gpu 0 &


    echo -e "\e[32mFederated batch effect correction of $dataset.\e[0m"
    output_path="${output}/federated"
    mkdir -p "$output_path"
    fedscgen-federated \
      --init_model_path "$init_model_path" \
      --adata "${prep_for_be_datapath}" \
      --output "$output_path" \
      --epoch 2 \
      --batch_key "batch_group" \
      --cell_key "${celltype_key}" \
      --batches "$BATCHES" \
      --lr 0.001 \
      --batch_size 50 \
      --hidden_size "800,800" \
      --z_dim 10 \
      --early_stopping_kwargs '{"early_stopping_metric": "val_loss", "patience": 20, "threshold": 0, "reduce_lr": True, "lr_patience": 13, "lr_factor": 0.1}' \
      --batch_out "0" \
      --remove_cell_types "" \
      --n_clients "${n_clients}" \
      --gpu 1 \
      --n_rounds 8 \
      --aggregation "fedavg" \
      --smpc

    wait

    echo -e "\e[32mPost processing the corrected data\e[0m"


    echo -e "\e[32mPost processing the centralized corrected data\e[0m"
    corrected_data_path="${output}/centralized/corrected.h5ad"
    python prep_batch_effect_correction.py \
            --prep_type "post" \
            --dataset "${ds_name}" \
            --corrected_data_path "${corrected_data_path}" \
            --celltype_key "${celltype_key}" \
            --reference_file "${reference_file_prefix}_corrected.h5ad" \
            --query_file "${query_file_prefix}_corrected.h5ad"


    echo -e "\e[32mPost processing the federated corrected data\e[0m"
    corrected_data_path="${output}/federated/fed_corrected.h5ad"
    python prep_batch_effect_correction.py \
            --prep_type "post" \
            --dataset "${ds_name}" \
            --corrected_data_path "${corrected_data_path}" \
            --celltype_key "${celltype_key}" \
            --reference_file "${reference_file_prefix}_fed_corrected.h5ad" \
            --query_file "${query_file_prefix}_fed_corrected.h5ad"
done
