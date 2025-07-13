#!/bin/bash

source ./configs.sh
# Assign command-line arguments to variables
datasetnames="${1}"
mode="${2}"
smpc="${3-false}"
GPU="${4-0}"

resolve_dataset_keys "$datasetnames"
# Make the annotation.sh script executable
chmod +x embedding.sh

echo "Running reference mapping for ${mode}"

# Loop through each dataset configuration
for key in "${keys[@]}"; do
    echo -e "\e[34mRunning reference mapping for $key\e[0m"

    # Read dataset configuration into an array using the custom delimiter
    IFS='|' read -r -a args <<< "${datasets[$key]}"
    echo "Arguments: ${args[0]} ${args[1]} ${args[2]} ${args[3]} ${args[4]} ${args[5]}"

    # Call the annotation.sh script with the appropriate arguments
    # order: mode, data_folder, adata_file, test_adata_file, celltype_key, batch_key, gpu, SMPC
    ./embedding.sh "${mode}" "${args[0]}" "${args[1]}" "${args[2]}" "${args[3]}" "${args[4]}" "${args[5]}" "${GPU}" "${smpc}"

    # Check for errors in the execution of the script
    if [ $? -ne 0 ]; then
        echo -e "\e[31mError processing dataset $key. Please check the configuration.\e[0m"
        continue
    fi
done

echo -e "\e[32mReference mapping completed for all datasets.\e[0m"

