#!/bin/bash

chmod +x run_param_tuning.sh
# By default on GPU:1
./run_param_tuning.sh &


# All following by default on GPU:0
chmod +x run_annotation.sh
./run_annotation.sh centralized_finetune_inference 20

./run_annotation.sh centralized_clients 20

chmod +x run_embedding.sh
./run_embedding.sh centralized &
./run_embedding.sh federated_zeroshot &
./run_embedding.sh centralized_clients &

wait

chmod +x run_perturbation.sh
# arguments: mode, reverse, n_epochs, n_rounds, GPU, n_clients, per_round_eval
./run_perturbation.sh centralized False 15 0 0 0 False &
./run_perturbation.sh centralized_clients False 15 0 1 2 False &
wait

# reverse
./run_perturbation.sh centralized True 15 0 0 0 False &
./run_perturbation.sh centralized_clients True 15 0 1 2 False &
wait

# Federated
./run_perturbation.sh federated_finetune False 3 10 0 2 True &
./run_perturbation.sh federated_finetune True 3 10 1 2 True &
wait

