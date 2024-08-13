#!/bin/bash

chmod +x run_param_tuning.sh
./run_param_tuning.sh

chmod +x run_annotation.sh
./run_annotation.sh centralized_finetune_inference 20

./run_annotation.sh centralized_clients 20