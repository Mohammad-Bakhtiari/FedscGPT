#!/bin/bash

chmod +x run_param_tuning.sh
./run_param_tuning.sh

chmod +x run_annotation.sh
./run_annotation.sh centralized_finetune_inference 20

./run_annotation.sh centralized_clients 20

chmod +x run_embedding.sh
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running embedding for all modes\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34mRunning embedding for centralized\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_embedding.sh centralized
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34mRunning embedding for federated without SMPC\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_embedding.sh federated_zeroshot
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34mRunning embedding for federated with SMPC\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_embedding.sh federated_zeroshot true
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34mRunning embedding for clients local training\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_embedding.sh centralized_clients