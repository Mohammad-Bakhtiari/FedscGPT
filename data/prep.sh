#!/bin/bash
data_dir="$1"
correction="${2:-false}"
correction_stage="${3:-uncorrected}" # options: uncorrected, corrected



if [[ "$correction" == "true" ]]; then
  if [[ "$correction_stage" == "uncorrected" ]]; then
    declare -A datasets
    datasets["COVID"]="covid|Covid_annot.h5ad|Covid_annot-uncorrected.h5ad|celltype|study"
    for key in "${!datasets[@]}"; do
      echo -e "\e[32mPreprocessing data for batch effect correction for ${key}\e[0m"

      # Parse the configuration string into an array
      IFS='|' read -r -a args <<< "${datasets[$key]}"

      python prep_batch_effect_correction.py \
        --data_dir "$data_dir/${args[0]}" \
        --orig_adata "${args[1]}" \
        --uncorrected_adata "${args[2]}" \
        --celltype_key "${args[3]}" \
        --batch_key "${args[4]}" \
        --stage "uncorrected"

    done
  elif [[ "$correction_stage" == "corrected" ]]; then
    declare -A datasets
    # Order: subdir|orig_file|corrected_adata|uncorrected_adata|reference_filename|query_filename|celltype_key|batch_key|query_batch|normalize|min_max
    datasets["COVID-cent"]="covid|Covid_annot-uncorrected.h5ad|corrected.h5ad|reference_corrected.h5ad|query_corrected.h5ad|celltype|q"
    datasets["COVID-fed"]="covid|Covid_annot-uncorrected.h5ad|fed_corrected.h5ad|reference_fed_corrected.h5ad|query_fed_corrected.h5ad|celltype|q"

    for key in "${!datasets[@]}"; do
      echo -e "\e[32mPreprocessing data after batch effect correction for ${key}\e[0m"
      IFS='|' read -r -a args <<< "${datasets[$key]}"
      python prep_batch_effect_correction.py \
        --data_dir "$data_dir/${args[0]}" \
        --uncorrected_adata "${args[1]}" \
        --corrected_adata "${args[2]}" \
        --reference_file "${args[3]}" \
        --query_file "${args[4]}" \
        --celltype_key "${args[5]}" \
        --stage "corrected"
  else
    echo -e "\e[31mInvalid correction stage: $correction_stage. Use 'uncorrected' or 'corrected'.\e[0m"
    exit 1
  fi

else
  echo -e "\e[32mPreprocessing data without batch effect correction\e[0m"
  script_name="prep.py"
  declare -A datasets
  datasets["CellLine"]="cl|CellLine.h5ad|reference.h5ad|query.h5ad|cell_type|batch|2|true|min_max"
  datasets["COVID"]="covid|Covid_annot.h5ad|reference_annot.h5ad|query_annot.h5ad|celltype|ref-query-split|q|true|min_max"
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
fi