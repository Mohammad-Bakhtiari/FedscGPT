#!/bin/bash
GPU=0

#chmod +x run_param_tuning.sh
## Arguments: datasetnames, aggregation method, weighted, smpc, N_ROUNDS, GPU, epochs_values
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34m Running parameter tuning for FedscGPT with SMPC\e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./run_param_tuning.sh "HP,MYELOID-top4+rest,MS" fedavg true true 20 $GPU 1-5
#
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34m Running parameter tuning for FedscGPT without SMPC\e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./run_param_tuning.sh "HP,MYELOID-top4+rest,MS" fedavg true false 20 $GPU 1-5
#
#
#chmod +x run_annotation.sh
## Arguments datasetnames, mode, n_epochs, n_rounds, smpc, GPU, agg_method, weighted, mu
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34m Running centralized annotation with scGPT\e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./run_annotation.sh "LUNG,CellLine,COVID,COVID-cent_corrected,COVID-fed-corrected" centralized_finetune_inference 20 0 false $GPU
#
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34m Running local clients annotation with scGPT\e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./run_annotation.sh "LUNG,CellLine,COVID,COVID-cent_corrected,COVID-fed-corrected" centralized_clients 20 0 false $GPU
#
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34m Running Federated annotation with FedscGPT using weighted FedAvg\e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./run_annotation.sh "LUNG,CellLine,COVID,COVID-cent_corrected,COVID-fed-corrected" federated_finetune 1 20 false $GPU fedavg true
#
#
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34mRunning Federated annotation with FedscGPT using weighted FedAvg and SMPC\e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./run_annotation.sh "LUNG,CellLine,COVID,COVID-cent_corrected,COVID-fed-corrected" federated_finetune 1 20 true $GPU fedavg true
#
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34m Running Federated annotation with FedscGPT using weighted FedProx aggregation \e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./run_annotation.sh "LUNG,CellLine,COVID,COVID-cent_corrected,COVID-fed-corrected" federated_finetune 1 20 false $GPU fedprox true 0.01
#
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34m Running Federated annotation with FedscGPT using weighted FedProx aggregation and SMPC \e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./run_annotation.sh "LUNG,CellLine,COVID,COVID-cent_corrected,COVID-fed-corrected" federated_finetune 1 20 true $GPU fedprox true 0.01


chmod +x run_embedding.sh
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34m Running embedding for all modes\e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34mRunning embedding for centralized\e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./run_embedding.sh all centralized false $GPU
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34mRunning embedding for federated without SMPC\e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./run_embedding.sh all federated_zeroshot false $GPU
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34mRunning embedding for federated with SMPC\e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./run_embedding.sh all federated_zeroshot true $GPU
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34mRunning embedding for clients local training\e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./run_embedding.sh all centralized_clients false $GPU
./run_embedding.sh  "HP,MYELOID-top4+rest,MS,LUNG,CellLine" centralized false 1
./run_embedding.sh  "HP,MYELOID-top4+rest,MS,LUNG,CellLine" centralized_clients false 1
./run_embedding.sh  "HP,MYELOID-top4+rest,MS,LUNG,CellLine" federated_zeroshot false 1
./run_embedding.sh  "HP,MYELOID-top4+rest,MS,LUNG,CellLine" federated_zeroshot true 1