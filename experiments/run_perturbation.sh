#!/bin/bash

# Assign command-line arguments to variables
mode="$1"
reverse="$2"
n_epochs="${3}"
n_rounds="${4}"
GPU="${5-0}"
n_clients="${6-2}"
per_round_eval="${7-False}"

# Validate reverse argument
if [[ "$reverse" != "True" && "$reverse" != "False" ]]; then
    echo -e "\e[31mError: reverse must be either 'True' or 'False' but reverse=${reverse}.\e[0m"
    exit 1
fi

# Validate per_round_eval argument
if [[ "$per_round_eval" != "True" && "$per_round_eval" != "False" ]]; then
    echo -e "\e[31mError: per_round_eval must be either 'True' or 'False' but per_round_eval=${per_round_eval}.\e[0m"
    exit 1
fi

# Make the annotation.sh script executable
chmod +x perturbation.sh

# Declare an associative array to store dataset configurations
declare -A datasets

# Add dataset configurations
# format: name|adata_file|test_adata_file|pert_to_plot
reverse_msg=""
if [ "${reverse}" == False ]; then
    datasets["ADAMSON"]="adamson|KCTD16+ctrl"
else
  reverse_msg="reverse"
fi
#datasets["NORMAN"]="norman|SAMD1+ZBTB1"


echo "Running ${reverse_msg} perturbation modelling for ${mode} mode round ${n_rounds} with ${n_epochs} epochs"

# Loop through each dataset configuration
for key in "${!datasets[@]}"; do
    echo -e "\e[34mRunning ${reverse_msg} perturbation modelling for $key on GPU $GPU\e[0m"

    # Read dataset configuration into an array using the custom delimiter
    IFS='|' read -r -a args <<< "${datasets[$key]}"
    echo "Arguments: ${args[0]} ${args[1]} ${args[2]} ${args[3]} ${args[4]}"

    # Call the perturbation.sh script with the appropriate arguments
    # order: mode, data_folder, adata_file, test_adata_file, celltype_key, batch_key, gpu
    ./perturbation.sh "${mode}" "${reverse}" "${args[0]}" "${GPU}" "${args[1]}" "${n_clients}"  "${n_epochs}" "${n_rounds}" "${per_round_eval}"

    # Check for errors in the execution of the script
    if [ $? -ne 0 ]; then
        echo -e "\e[31mError in ${reverse_msg} perturbation for dataset $key.\e[0m"
        continue
    fi
done

echo -e "\e[33mData perturbation modelling completed for all datasets.\e[0m"
