#!/bin/bash


root_dir="$(dirname "$PWD")"
df_file="${root_dir}/output/annotation/param-tuning-res.csv"
pkl_file="${root_dir}/output/annotation/param-tuning-res.pkl"
data_dir="${root_dir}/data/benchmark"
# Plot centralized box-plots for accuracy
python3 plots.py --plot "cent_box_plt" --mode "centralized" --root_dir $root_dir  --param_tuning_df  $df_file \
--metric "accuracy"


# Plot heatmap for various metrics on dif rounds for various n_epochs
python3 plots.py --plot "metric_heatmap" --param_tuning_df $df_file

# Plot communication efficiency hist
python3 plots.py --plot "communication" --param_tuning_df $df_file

# Plot metric(Accuracy) changes over rounds and epochs
python3 plots.py --plot "accuracy_changes" --param_tuning_df $df_file

# Plot confusion matrices federated vs centralized for different datasets
python3 plots.py --plot "conf_matrix" --root_dir $root_dir --param_tuning_pkl $pkl_file --data_dir $data_dir --param_tuning_df $df_file