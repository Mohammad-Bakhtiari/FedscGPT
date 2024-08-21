#!/bin/bash

chmod +x run_param_tuning.sh
./run_param_tuning.sh

chmod +x run_annotation.sh
./run_annotation.sh centralized_finetune_inference 20

./run_annotation.sh centralized_clients 20

chmod +x run_embedding.sh
./run_embedding.sh centralized
./run_embedding.sh federated_zeroshot
./run_embedding.sh centralized_clients