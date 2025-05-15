#!/bin/bash


root_dir="$(dirname "$PWD")"
param_tuning_df_file="${root_dir}/output/annotation/param-tuning-res.csv"
param_tuning_pkl_file="${root_dir}/output/annotation/param-tuning-res.pkl"
# param_tuning_files expected to end with `-smpc` if smpc is true
data_dir="${root_dir}/data/scgpt/benchmark"
results_dir="${root_dir}/analysis/plots"
logfile="${results_dir}/plots.log"

# Redirect all output to the logfile
exec > >(tee -a "$logfile") 2>&1

if [ ! -d "${results_dir}" ]; then
    mkdir -p "${results_dir}"
fi
if [ ! -d "${results_dir}/annotation" ]; then
    mkdir -p "${results_dir}/annotation"
fi
if [ ! -d "${results_dir}/embedding" ]; then
    mkdir -p "${results_dir}/embedding"
fi

# Plot centralized box-plots for accuracy
python3 plots.py --plot "annotation_cent_box_plt" --mode "centralized" --root_dir $root_dir --data_dir $data_dir --param_tuning_df  $param_tuning_df_file \
--metric "accuracy"


# Plot heatmap for various metrics on dif rounds for various n_epochs
python3 plots.py --plot "annotation_metric_heatmap" --param_tuning_df $param_tuning_df_file --format 'png'

# Plot communication efficiency hist
python3 plots.py --plot "annotation_communication" --param_tuning_df $param_tuning_df_file

## Plot metric(Accuracy) changes over rounds and epochs
python3 plots.py --plot "annotation_accuracy_changes" --param_tuning_df $param_tuning_df_file --format 'svg'

# Plot confusion matrices FedscGPT-SMPC (best params based on accuracy) vs scGPT for different datasets
python3 plots.py --plot "annotation_conf_matrix" --root_dir $root_dir --param_tuning_pkl $param_tuning_pkl_file --data_dir $data_dir --param_tuning_df $param_tuning_df_file

# Plot best metrics based on each metric
python3 plots.py --plot "annotation_best_metrics" --root_dir "${root_dir}/output/annotation" --param_tuning_df $param_tuning_df_file --format 'png'

# Plot reference mapping boxplot
python3 plots.py --plot "reference_map_boxplot" --root_dir "${root_dir}/output/embedding" --format 'svg'

python3 plots.py --plot "fed_embedding_umap" --root_dir "${root_dir}/output/embedding" --data_dir $data_dir