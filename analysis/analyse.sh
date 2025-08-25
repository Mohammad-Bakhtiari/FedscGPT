#!/bin/bash


root_dir="$(dirname "$PWD")"
results_df_file="${root_dir}/output/annotation/results_summary.csv"
results_pkl_file="${root_dir}/output/annotation/results_summary.pkl"
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

plot_scenarios=(
  "investigate"
  "fig1e_figs2_annot_scatter_plot"
  "figs3_figs4_heatmap"
  "fig1_fig3_figs1_figs5_umap_conf_matrix"
  "fig2_comm_efficiency"
  "fig4-reference_map_boxplot"
  "myeloid"
  "covid"
)

for plot in "${plot_scenarios[@]}"; do
  echo ">>> Generating plot: $plot"
  python3 plots.py \
    --plot "$plot" \
    --root_dir "$root_dir" \
    --data_dir "$data_dir" \
    --final_df "$results_df_file" \
    --final_pkl "$results_pkl_file" \
    --format "png"
done


