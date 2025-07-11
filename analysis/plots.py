import pandas as pd

import __init__
import os
import argparse
from analysis.utils import (CentralizedMetricPlotter, collect_cent_metrics, plot_tuning_heatmap, find_best_fed,
                            plot_communication_efficiency, plot_metric_cahnges_over_ER, plot_umap_and_conf_matrix,
                            plot_best_metrics, embedding_boxplot, fed_embedding_umap, accuracy_annotated_scatterplot,
                            plot_batch_effect_umaps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, choices=["annotation_cent_box_plt",
                                                     "annotation_metric_heatmap",
                                                     'annotation_communication',
                                                     'annotation_accuracy_changes',
                                                     'annotation_conf_matrix',
                                                     'annotation_best_metrics',
                                                     'annotation_cent_box_plt',
                                                     'reference_map_boxplot',
                                                     'fed_embedding_umap',
                                                     "umap_covid_batch_effect",
                                                     "umap_ms_batch_effect",
                                                     ], default='annotation_cent_box_plt')
    parser.add_argument("--mode", choices=['centralized', 'federated'], default='centralized')
    parser.add_argument("--root_dir", type=str, default='/home/bba1658/FedscGPT/output/annotation')
    parser.add_argument("--metric", choices=['accuracy', 'precision', 'recall', 'macro_f1'], default="accuracy")
    parser.add_argument("--param_tuning_df", type=str, default='/home/bba1658/FedscGPT/output/annotation/param_tuning_res.csv')
    parser.add_argument("--param_tuning_pkl", type=str,
                        default='/home/bba1658/FedscGPT/output/annotation/param_tuning_res.pkl')
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default='svg')
    parser.add_argument("--data_dir", type=str, default='/home/bba1658/FedscGPT/data/benchmark')
    args = parser.parse_args()
    param_tuning_smpc_df = args.param_tuning_df.replace('.csv', '-smpc.csv')
    param_tuning_smpc_pkl = args.param_tuning_pkl.replace('.pkl', '-smpc.pkl')
    if args.plot == 'annotation_cent_box_plt': # Probably not used
        metrics = {}
        print("[FedscGPT]: Best results found ")
        fedscgpt = find_best_fed(args.param_tuning_df, args.metric)
        print("[FedscGPT-SMPC]: Best results found ")
        fedscgpt_smpc = find_best_fed(param_tuning_smpc_df, args.metric)
        for dataset in ["hp", "ms", "myeloid"]:
            metrics[dataset] = collect_cent_metrics("/".join([args.root_dir, "output", "annotation", dataset, 'centralized']),
                                                    args.data_dir, args.metric)
            metrics[dataset]['FedscGPT-SMPC'] = fedscgpt_smpc[dataset]
            metrics[dataset]['FedscGPT'] = fedscgpt[dataset]
        plotter = CentralizedMetricPlotter()
        df = plotter.collect_data(metrics)
        df.to_csv('clients_cent.csv')
        df = pd.read_csv('clients_cent.csv')
        df["Metric"] = "Accuracy"
        df["Value"] = df["Accuracy"]
        df.drop(columns=["Accuracy"], inplace=True)
        accuracy_annotated_scatterplot(df, "./plots/annotation", args.format)
    elif args.plot == 'annotation_metric_heatmap':
        # Old: Supplementary Figure 2, New: two figures for FedscGPT with or without SMPC
        plot_tuning_heatmap(args.param_tuning_df, plot_name="metrics_heatmap", file_format=args.format)
        plot_tuning_heatmap(param_tuning_smpc_df, plot_name="metrics_heatmap-smpc", file_format=args.format)
    elif args.plot == 'annotation_communication':
        # Old: Figure 2d, New: Figure 3
        plot_communication_efficiency(args.param_tuning_df, param_tuning_smpc_df)
    elif args.plot == 'annotation_accuracy_changes':
        # Old: Figure 2c, New: Figure 3
        plot_metric_cahnges_over_ER(param_tuning_smpc_df, img_format=args.format)
    elif args.plot == 'annotation_conf_matrix':
        # Old: UMAPS, legends, and conf Figures 2a-b, supplementary 3 and 4
        # New: Figure 2a-d, and Supplementary x
        plot_umap_and_conf_matrix(args.root_dir, args.data_dir, param_tuning_smpc_pkl, param_tuning_smpc_df)
    elif args.plot == 'annotation_best_metrics':
        # Old: Supplementary figure 5, New: Figure 3
        plot_best_metrics(args.root_dir, args.param_tuning_df, img_format=args.format)
    elif args.plot == 'reference_map_boxplot':
        # Old: Figure 3, New: Figure 4a
        embedding_boxplot(args.root_dir, args.format)
    elif args.plot == "fed_embedding_umap":
        fed_embedding_umap(args.data_dir, args.root_dir, img_format='png')
    elif args.plot == "umap_covid_batch_effect":
        plot_batch_effect_umaps(
            raw_h5ad = os.path.join(args.data_dir, "preprocessed_for_be.h5ad"),
            cent_corrected=os.path.join(args.data_dir, "output", "centralized", "corrected.h5ad"),
            fed_corrected=os.path.join(args.data_dir, "output", "federated", "fed_corrected.h5ad"),
            batch_key = "batch_group",
            cell_key = "celltype",
            out_prefix = os.path.join(args.data_dir, "plots")
        )
    elif args.plot == "umap_ms_batch_effect":
        plot_batch_effect_umaps(
            raw_h5ad = os.path.join(args.data_dir, "refined_ms.h5ad"),
            cent_corrected = os.path.join(args.data_dir, "output", "centralized", "corrected.h5ad"),
            fed_corrected = os.path.join(args.data_dir, "output", "federated", "fed_corrected.h5ad"),
            batch_key = "Factor Value[sampling site]",
            cell_key = "Factor Value[inferred cell type - authors labels]",
            out_prefix = os.path.join(args.data_dir, "umap_ms_batch_effect")
        )
