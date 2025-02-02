import pandas as pd

import __init__
import argparse
from analysis.utils import (CentralizedMetricPlotter, collect_metrics, plot_tuning_heatmap, find_best_fed,
                            analyze_communication_efficiency, plot_metric_cahnges_over_ER, plot_umap_and_conf_matrix,
                            plot_best_metrics, embedding_boxplot)

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
    if args.plot == 'annotation_cent_box_plt':
        base_paths = ["/".join([args.root_dir, ds, args.mode]) for ds in ["hp", "ms", "myeloid"]]
        metrics = {}
        best = find_best_fed(args.param_tuning_df, args.metric)
        for dataset in ["hp", "ms", "myeloid"]:
            metrics[dataset] = collect_metrics("/".join([args.root_dir, "output", "annotation", dataset, args.mode]), args.metric)
            metrics[dataset]['federated'] = best[dataset]
        plotter = CentralizedMetricPlotter()
        df = plotter.collect_data(metrics)
        df.to_csv('clients_cent.csv')
        df = pd.read_csv('clients_cent.csv')
        plotter.plot_data_matplotlib(df, 'Accuracy', 'centralized_metric_plot', 'svg')
    elif args.plot == 'annotation_metric_heatmap':
        plot_tuning_heatmap(args.param_tuning_df, plot_name="metrics_heatmap", file_format=args.format)
    elif args.plot == 'annotation_communication':
        analyze_communication_efficiency(args.param_tuning_df, 'clients_cent.csv' )
    elif args.plot == 'annotation_accuracy_changes':
        plot_metric_cahnges_over_ER(args.param_tuning_df, img_format=args.format)
    elif args.plot == 'annotation_conf_matrix':
        plot_umap_and_conf_matrix(args.root_dir, args.data_dir, args.param_tuning_pkl, args.param_tuning_df)
    elif args.plot == 'annotation_best_metrics':
        plot_best_metrics(args.root_dir, args.param_tuning_df, img_format=args.format)
    elif args.plot == 'reference_map_boxplot':
        embedding_boxplot(args.root_dir, args.format)
