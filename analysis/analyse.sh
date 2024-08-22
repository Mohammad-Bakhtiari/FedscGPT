#!/bin/bash


root_dir="$(dirname "$PWD")"
df_file="${root_dir}/output/annotation/param-tuning-res.csv"
pkl_file="${root_dir}/output/annotation/param-tuning-res.pkl"
data_dir="${root_dir}/data/benchmark"
results_dir="plots"
logfile="${results_dir}/plots.log"

# Redirect all output to the logfile
exec > >(tee -a "$logfile") 2>&1

if [ ! -d "${results_dir}" ]; then
    mkdir -p "${results_dir}"
fi

# Plot centralized box-plots for accuracy
python3 plots.py --plot "annotation_cent_box_plt" --mode "centralized" --root_dir $root_dir  --param_tuning_df  $df_file \
--metric "accuracy"


# Plot heatmap for various metrics on dif rounds for various n_epochs
python3 plots.py --plot "annotation_metric_heatmap" --param_tuning_df $df_file --format 'png'

# Plot communication efficiency hist
python3 plots.py --plot "annotation_communication" --param_tuning_df $df_file

## Plot metric(Accuracy) changes over rounds and epochs
python3 plots.py --plot "annotation_accuracy_changes" --param_tuning_df $df_file --format 'svg'

# Plot confusion matrices federated vs centralized for different datasets
python3 plots.py --plot "annotation_conf_matrix" --root_dir $root_dir --param_tuning_pkl $pkl_file --data_dir $data_dir --param_tuning_df $df_file

# Plot best metrics
python3 plots.py --plot "annotation_best_metrics" --root_dir "${root_dir}/output/annotation" --param_tuning_df $df_file --format 'png'

# Plot reference mapping boxplot
python3 plots.py --plot "reference_map_boxplot" --root_dir "${root_dir}/output/embedding" --format 'svg'