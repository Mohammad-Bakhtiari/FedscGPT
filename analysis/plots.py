import pandas as pd
import anndata

import __init__
import os
import argparse
from analysis.utils import (plot_metric_changes_over_ER, plot_umap_and_conf_matrix, embedding_boxplot,
                            accuracy_annotated_scatterplot, investigate_general, investigate_detailed,
                            summarize_best_hyperparams_by_metric, communication_efficiency_table,
                            plot_stacked_fedavg_heatmaps, plot_epoch_round_per_mu, export_optimal_params_by_metric,
                            plot_best_fed_vs_centralized, plot_myeloid_over_rounds,
                            plot_best_fed_myeloid_vs_centralized, plot_myeloid_over_rounds_vs_central,
                            plot_best_per_metric_covid, covid_scatterplot, best_perf_table,
                            best_perf_table_reference_mapping, myeloid_scatterplot, plot_batch_effect,
                            filter_and_export_metrics
                            )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, choices=[
        "investigate",
        "fig1e_figs2_annot_scatter_plot",
        "figs3_figs4_heatmap",
        "fig1_fig3_figs1_figs5_umap_conf_matrix",
        "fig2_comm_efficiency",
        'fig4-reference_map_boxplot',
        "myeloid",
        "covid",


        ], default='annotation_cent_box_plt')
    parser.add_argument("--mode", choices=['centralized', 'federated'], default='centralized')
    parser.add_argument("--root_dir", type=str, default='/home/bba1658/FedscGPT/output/annotation')
    parser.add_argument("--metric", choices=['accuracy', 'precision', 'recall', 'macro_f1'], default="accuracy")
    parser.add_argument("--final_df", type=str)
    parser.add_argument("--final_pkl", type=str)
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default='svg')
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    df = pd.read_csv(args.final_df)
    if args.plot == 'investigate':
        investigate_general(df)
        investigate_detailed(df)
        summarize_best_hyperparams_by_metric(df)
        communication_efficiency_table(df)
        best_perf_table(df)
    elif args.plot == 'figs3_figs4_heatmap':
        plot_stacked_fedavg_heatmaps(df)
        plot_epoch_round_per_mu(df)
    elif args.plot == 'fig1e_figs2_annot_scatter_plot':
        accuracy_annotated_scatterplot(df, plots_dir="./plots/fig1e", img_format="png")
    elif args.plot == "fig1_fig3_figs1_figs5_umap_conf_matrix":
        summary = pd.read_pickle(args.final_pkl)
        plot_umap_and_conf_matrix(df, summary, args.data_dir)
    elif args.plot == "fig2_comm_efficiency":
        xls = pd.ExcelFile("communication_efficiency_summary.xlsx")
        dataset_name_map = {
            "hp5": "HP",
            "ms": "MS",
            "myeloid-top5": "Myeloid"
        }
        filter_and_export_metrics(xls, "communication_efficiency_metrics.xlsx", dataset_name_map)
        plot_metric_changes_over_ER(df, dataset_name_map, [1, 2, 3, 4, 5], img_format=args.format)
        xls = pd.ExcelFile("best_hyperparams_per_federated_setting.xlsx")
        dataset_name_map.update({'lung': 'Lung', 'cl': 'CL'})
        export_optimal_params_by_metric(xls, dataset_name_map)
        df = df[df['Dataset'].isin(dataset_name_map.keys())].copy()
        df['Dataset'] = df['Dataset'].map(dataset_name_map)
        plot_best_fed_vs_centralized(df)
    elif args.plot == 'fig4-reference_map_boxplot':
        dataset_name_map = {'hp5': 'HP', 'lung': 'Lung-Kim', 'cl': 'CL', 'ms': 'MS', 'myeloid-top5': 'Top5',
                            'myeloid-top10': 'Top10', 'myeloid-top20': 'Top20', 'myeloid-top30': 'Top30',
                            'covid': 'Uncorrected', 'covid-corrected': 'Corrected', 'covid-fed-corrected': 'Fed-Corrected'}
        df = pd.read_csv("/home/mohammad/PycharmProjects/FedscGPT/output/embedding/all_eval_metrics.csv")
        df.rename(columns={
            'macro_f1': 'Macro-F1',
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall'
        }, inplace=True)
        best_perf_table_reference_mapping(df)
        df['Dataset'] = df['Dataset'].map(dataset_name_map)
        df_datasets = df[df['Dataset'].isin(dataset_name_map.values())].copy()
        embedding_boxplot(df_datasets, ['HP', 'Lung-Kim', 'CL', 'MS'], "datasets", args.format)
        df_myeloid = df[df['Dataset'].str.startswith('Top')].copy()
        embedding_boxplot(df_myeloid, ['Top5', 'Top10', 'Top20', 'Top30'], "myeloid", args.format)
        df_covid = df[df['Dataset'].isin(['Uncorrected', 'Corrected'])].copy()
        embedding_boxplot(df_covid, ['Uncorrected', 'Corrected'], "covid", args.format)

    elif args.plot == 'myeloid':
        dataset_name_map = {'myeloid-top5': 'Top5',
                            'myeloid-top10': 'Top10', 'myeloid-top20': 'Top20', 'myeloid-top30': 'Top30',}
        df = df[df['Dataset'].isin(dataset_name_map.keys())].copy()
        df['Dataset'] = df['Dataset'].map(dataset_name_map)
        df.Metric = df.Metric.apply(lambda x: 'Macro-F1' if x == 'Macro_F1' else x)
        df.reset_index(drop=True, inplace=True)
        plot_best_fed_myeloid_vs_centralized(df, subdir="myeloid")
        plot_myeloid_over_rounds_vs_central(df, subdir="myeloid")
        plot_myeloid_over_rounds(df, subdir="myeloid")
        myeloid_scatterplot(df, plots_dir="./plots/annotation/myeloid", img_format="png")
    elif args.plot == 'covid':
        dataset_name_map = {'covid': 'Uncorrected', 'covid-corrected': 'Corrected'}
        df = df[df['Dataset'].isin(dataset_name_map.keys())].copy()
        df['Dataset'] = df['Dataset'].map(dataset_name_map)
        df.Aggregation = df['Aggregation'].apply(lambda x: x.replace('weighted-', '') if 'weighted-' in x else x)
        df.Aggregation = df['Aggregation'].apply(lambda x: f"{x[5:]}-SMPC" if x.startswith('SMPC-') else x)
        print(df.Aggregation.unique())
        print(df)
        plot_best_per_metric_covid(df, subdir="covid")
        covid_scatterplot(df, plots_dir="./plots/annotation/covid")
        plot_batch_effect('covid', args.data_dir)
        plot_batch_effect('covid-corrected', args.data_dir)

