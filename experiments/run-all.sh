#!/bin/bash
GPU=0

chmod +x run_param_tuning.sh
# Arguments: datasetnames, aggregation method, weighted, smpc, N_ROUNDS, GPU, epochs_values
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running parameter tuning for FedscGPT with SMPC\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_param_tuning.sh "HP,MYELOID-top4+rest" fedavg true true 20 $GPU

echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running parameter tuning for FedscGPT without SMPC\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_param_tuning.sh "HP,MYELOID-top4+rest" fedavg true false 20 $GPU

echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running mu tuning for FedscGPT with SMPC and FedProx \e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_param_tuning.sh "MS-cent_corrected,COVID-cent_corrected" fedprox true true 20 $GPU "1,2,3,4,5"


chmod +x run_annotation.sh
# Arguments datasetnames, mode, n_epochs, n_rounds, smpc, GPU, agg_method, weighted, mu
echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running centralized annotation with scGPT\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_annotation.sh all centralized_finetune_inference 20 0 false $GPU

echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running local clients annotation with scGPT\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_annotation.sh all centralized_clients 20 0 false $GPU

echo -e "\e[34m-------------------------------------\e[0m"
echo -e "\e[34m Running Federated annotation with FedscGPT\e[0m"
echo -e "\e[34m-------------------------------------\e[0m"
./run_annotation.sh "HP,MYELOID-top4+rest" federated_finetune 1 20 true $GPU fedavg true 0.01



chmod +x annotation.sh
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34m Running annotation with FedscGPT without SMPC on MS dataset for two local epochs and seven rounds\e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./annotation.sh federated_finetune ms reference.h5ad query.h5ad "Factor Value[inferred cell type - authors labels]" "Factor Value[sampling site]" $GPU 2 7 false
#
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34m Running annotation with FedscGPT without SMPC on HP dataset for one local epoch and three rounds\e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./annotation.sh federated_finetune hp reference_refined.h5ad query.h5ad Celltype batch $GPU 1 3 false
#
#echo -e "\e[34m-------------------------------------\e[0m"
#echo -e "\e[34m Running annotation with FedscGPT without SMPC on Myeloid dataset for four local epochs and one round\e[0m"
#echo -e "\e[34m-------------------------------------\e[0m"
#./annotation.sh federated_finetune myeloid reference_adata.h5ad query_adata.h5ad combined_celltypes top4+rest $GPU 4 1 false


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