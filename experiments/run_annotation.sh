#!/bin/bash

# Assign command-line arguments to variables
mode="$1"
n_epochs="${2}"
n_rounds="${3}"

# Make the annotation.sh script executable
chmod +x annotation.sh

# Declare an associative array to store dataset configurations
declare -A datasets

GPU=0

# Add dataset configurations
# format: data_folder|adata_file|test_adata_file|celltype_key|batch_key|gpu
datasets["MS"]="ms|reference.h5ad|query.h5ad|Factor Value[inferred cell type - authors labels]|Factor Value[sampling site]"
datasets["HP"]="hp|reference_refined.h5ad|query.h5ad|Celltype|batch"
datasets["MYELOID-top4+rest"]="myeloid|reference_adata.h5ad|query_adata.h5ad|combined_celltypes|top4+rest"

echo "Running annotation for ${mode}"

# Loop through each dataset configuration
for key in "${!datasets[@]}"; do
    echo "Running data annotation for $key"

    # Read dataset configuration into an array using the custom delimiter
    IFS='|' read -r -a args <<< "${datasets[$key]}"
    echo "Arguments: ${args[0]} ${args[1]} ${args[2]} ${args[3]} ${args[4]} ${args[5]}"

    # Call the annotation.sh script with the appropriate arguments
    # order: mode, data_folder, adata_file, test_adata_file, celltype_key, batch_key, gpu
    ./annotation.sh "${mode}" "${args[0]}" "${args[1]}" "${args[2]}" "${args[3]}" "${args[4]}" "${GPU}" "${n_epochs}" "${n_rounds}"

    # Check for errors in the execution of the script
    if [ $? -ne 0 ]; then
        echo "Error processing dataset $key. Please check the configuration."
        continue
    fi
done

echo "Data annotation completed for all datasets."
