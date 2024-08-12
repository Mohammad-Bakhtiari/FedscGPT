#!/bin/bash


root_dir="$(dirname "$PWD")"


# Plot centralized box-plots for accuracy
python plots.py --plot "cent_box_plt" --mode "centralized" --root_dir "${root_dir}/output/annotation" --metric "accuracy"


# Plot heatmap for various metrics on dif rounds for various n_epochs
python plots.py --plot "metric_heatmap" --filepath "${root_dir}/output/annotation/shared_res.csv"

