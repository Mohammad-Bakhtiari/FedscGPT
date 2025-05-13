#!/bin/bash

chmod +x run_param_tuning.sh
./run_param_tuning.sh

chmod +x run_annotation.sh
./run_annotation.sh centralized_finetune_inference 20

./run_annotation.sh centralized_clients 20

chmod +x run_embedding.sh
echo "-------------------------------------"
echo "Running embedding for all modes"
echo "-------------------------------------"
echo "Running embedding for centralized"
echo "-------------------------------------"
./run_embedding.sh centralized
echo "-------------------------------------"
echo "Running embedding for federated without SMPC"
echo "-------------------------------------"
./run_embedding.sh federated_zeroshot
echo "-------------------------------------"
echo "Running embedding for federated with SMPC"
echo "-------------------------------------"
./run_embedding.sh federated_zeroshot true
echo "-------------------------------------"
echo "Running embedding for clients local training"
echo "-------------------------------------"
./run_embedding.sh centralized_clients