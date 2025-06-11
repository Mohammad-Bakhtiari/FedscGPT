import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.metrics import confusion_matrix
import plotly.express as px
from PIL import Image
import numpy as np
import anndata
import scanpy as sc
import random

from FedscGPT.utils import print_config, generate_palette

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

set_seed(SEED)
image_format = 'svg'
ANNOTATION_PLOTS_DIR = 'plots/annotation'
FEDSCGPT_MARKER = 'D'
FEDSCGPT_SMPC_MARKER = '*'


def load_metric(filepath, metric):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data['results'][f'test/{metric}']

def collect_cent_metrics(base_path, data_dir, metric):
    accuracies = {}
    for root, dirs, files in os.walk(base_path):
        if 'results.pkl' in files:
            client_name = os.path.basename(root)
            accuracy = load_metric(os.path.join(root, 'results.pkl'), metric)
            if client_name.startswith('client'):
                ds = root.split('/')[-3]
                h5ad_file_dir = os.path.join(data_dir, ds,client_name, 'adata.h5ad')
                if os.path.exists(h5ad_file_dir):
                    batch = get_clients_batch_value(h5ad_file_dir, ds)
                elif ds == "ms":
                    batch = client_name
                else:
                    raise ValueError(f'Dataset {ds} does not exist')
            else:
                batch = "scGPT"
            accuracies[batch] = accuracy
    return accuracies


class CentralizedMetricPlotter:

    @staticmethod
    def collect_data(metrics):
        """
        Collect data from metrics into a pandas DataFrame.
        """
        data = []
        for dataset, values in metrics.items():
            scgpt_acc = values.pop('scGPT')
            fedscgpt_acc = values.pop('FedscGPT')
            fedscgpt_smpc_acc = values.pop('FedscGPT-SMPC')
            # Append each client's data
            for client, acc in values.items():
                data.append({'Dataset': dataset, 'Type': client, 'Accuracy': acc})

            # Append centralized accuracy
            data.append({'Dataset': dataset, 'Type': 'scGPT', 'Accuracy': scgpt_acc})

            # Append federated accuracy
            data.append({'Dataset': dataset, 'Type': 'FedscGPT-SMPC', 'Accuracy': fedscgpt_smpc_acc})

            data.append({'Dataset': dataset, 'Type': 'FedscGPT', 'Accuracy': fedscgpt_acc})

        df = pd.DataFrame(data)
        return df

    @staticmethod
    def plot_data(df, metric_name, plot_name, img_format='svg'):
        """
        Plot data using Plotly from a pandas DataFrame.
        """
        # Create a boxplot for Client accuracies
        fig = px.box(df[df['Type'] == 'Client'], x='Dataset', y='Accuracy', points="all",
                     title=f"{metric_name.capitalize()} Distribution per Dataset",
                     labels={'Accuracy': metric_name.capitalize()})

        # Add centralized and federated accuracies
        for dataset in df['Dataset'].unique():
            centralized = df[(df['Dataset'] == dataset) & (df['Type'] == 'Centralized')]['Accuracy'].values[0]
            federated = df[(df['Dataset'] == dataset) & (df['Type'] == 'Federated')]['Accuracy'].values[0]

            fig.add_scatter(x=[dataset], y=[centralized], mode='lines+markers', name='Centralized Accuracy',
                            line=dict(color='red', dash='dash'))
            fig.add_scatter(x=[dataset], y=[federated], mode='markers', name='Federated Accuracy',
                            marker=dict(color='blue', symbol='circle'))

        # Save the plot
        fig.write_image(f"{ANNOTATION_PLOTS_DIR}/{plot_name}.{img_format}")

        # Show the plot
        fig.show()

        class HandlerImage:
            def __init__(self, image):
                self.image = image

            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                imagebox = OffsetImage(self.image, zoom=0.15)
                ab = AnnotationBbox(imagebox, (x0 + .075, y0 + .08), frameon=False, xycoords='axes fraction')
                handlebox.add_artist(ab)
                return ab

        plt.legend(handles=legend_elements, labels=legend_labels, loc='lower left', fontsize=14, labelspacing=0.9,
                   borderpad=.3, handlelength=1.4,
                   handler_map={image_placeholder_instance: HandlerImage(boxplot_image)})


        # Customize the plot
        plt.xlabel('Datasets', fontsize=16)
        plt.ylabel(metric_name.capitalize(), fontsize=16)
        plt.xticks(range(1, len(datasets) + 1), [handle_ds_name(d) for d in datasets], fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()

        # Save the plot
        if img_format == "svg":
            plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{plot_name}.svg", format='svg')
        else:
            plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{plot_name}.png")

def plot_tuning_heatmap(file_path, plot_name, file_format='png'):
    """
    Plot heatmap of various metrics from the results DataFrame.

    Args:
        file_path (str): Path to the CSV file containing the results.
        plot_name (str): Name for the saved plot file.
        file_format (str): Format to save the plot ('png', 'jpg', etc.).
    """
    # Load the results DataFrame
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df = df[~(df.Round == 0)]
    # Determine unique datasets and metrics
    dataset_keys = df['Dataset'].unique()
    metric_keys = df['Metric'].unique()

    fig, axs = plt.subplots(len(metric_keys), len(dataset_keys),
                            figsize=(len(dataset_keys) *7, len(df['n_epochs'].unique()) + 1),
                            squeeze=True)
    fig.subplots_adjust(left=0.08, right=0.8, top=0.9, bottom=0.15, wspace=0.01, hspace=0.01)

    norm = colors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('Reds_r')
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    for j, dataset in enumerate(dataset_keys):
        for i, metric in enumerate(metric_keys):
            ax = axs[i][j]
            data = df[(df['Dataset'] == dataset) & (df['Metric'] == metric)]

            # Find the index of the maximum value
            max_index = data['Value'].idxmax()
            best_epoch = data.loc[max_index, "n_epochs"]
            best_round = data.loc[max_index, "Round"]
            max_value_diff = data.loc[max_index, 'Value']

            pivot = data.pivot(index='n_epochs', columns='Round', values='Value')
            sns.heatmap(pivot, ax=ax, cmap=cmap, cbar=False, center=0, vmin=0, vmax=1, square=True)

            #
            ax.text(best_round - 0.5, best_epoch - 0.5, f"{max_value_diff:.2f}", ha='center', va='center', fontsize=12)
            ticklabels = [''] * 20
            ticklabels[0] = '1'
            for l in range(1, 21):
                if l % 5 == 0:
                    ticklabels[l - 1] = str(l)
            ax.set_xticklabels(ticklabels, fontsize=14)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
            ax.grid(False)
            if j == 0:
                ax.text(-0.22, 0.15, metric, transform=ax.transAxes, fontsize=16, va='bottom', ha='left',
                        rotation=90)
                ax.set_ylabel("Epochs", fontsize=14)
            else:
                ax.set_ylabel('')
                ax.yaxis.set_visible(False)
            if i == 0:
                ax.set_title(handle_ds_name(dataset), fontsize=14, loc='left')
            if i < len(metric_keys) - 1:
                ax.xaxis.set_visible(False)
            else:
                ax.set_xlabel("Rounds", fontsize=14)
    # Adjust layout for colorbar and legend
    plt.subplots_adjust(right=0.79)
    cbar_ax = fig.add_axes([0.79, 0.28, 0.02, 0.5])
    cbar = plt.colorbar(mappable, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)
    plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{plot_name}.{file_format}", dpi=300)


def plot_communication_efficiency(fedscgpt_param_tuning, fedscgpt_smpc_param_tuning):
    fedscgpt_table = analyze_communication_efficiency(fedscgpt_param_tuning, 'clients_cent.csv')
    fedscgpt_smpc_table = analyze_communication_efficiency(fedscgpt_smpc_param_tuning, 'clients_cent.csv', smpc=True)
    plot_communication_comparison(
        fedscgpt_table,
        fedscgpt_smpc_table,
        out_path=f"{ANNOTATION_PLOTS_DIR}/communication_comparison.svg"
    )


def analyze_communication_efficiency(results_file_path, centralized_file_path,
                                     percentages=[70, 80, 90, 95, 99],
                                     metric="Accuracy", smpc=False):
    """
    ...
    """
    df = pd.read_csv(results_file_path)
    df.dropna(inplace=True)
    df = df[df.Round != 0]
    centralized_df = pd.read_csv(centralized_file_path)

    approach_name = 'FedscGPT-SMPC' if smpc else 'FedscGPT'

    dataset_keys = df['Dataset'].unique()
    table_data = [["Dataset"] + [f"{p}%" for p in percentages]]

    for dataset in dataset_keys:
        row = [handle_ds_name(dataset)]
        for p in percentages:
            central_value = centralized_df[
                (centralized_df['Dataset'] == dataset) &
                (centralized_df['Type'] == 'scGPT')
            ][metric].values[0]

            data = df[(df['Dataset'] == dataset) & (df['Metric'] == metric)]
            target_value = central_value * (p / 100)

            rounds_needed = epochs_needed = None
            best_ = None
            max_rounds = int(data['Round'].max())

            for r in range(1, max_rounds + 1):
                rounds_data = data[data['Round'] == r]
                if rounds_data.empty:
                    continue
                idx = rounds_data['Value'].idxmax()
                val = rounds_data.loc[idx, 'Value']
                if val >= target_value:
                    rounds_needed = r
                    epochs_needed = int(rounds_data.loc[idx, 'n_epochs'])
                    best_ = val
                    break

            if rounds_needed is not None:
                row.append(f"{rounds_needed}|{epochs_needed}")
                print(
                    f"For {dataset}, {p}% of centralized accuracy "
                    f"({central_value:.3f}) is reached by {approach_name} "
                    f"in {rounds_needed} rounds and {epochs_needed} epochs "
                    f"with {best_:.3f}"
                )
            else:
                row.append("NR")

        table_data.append(row)
    return table_data

import matplotlib.pyplot as plt

def plot_communication_comparison(fedscgpt_table, fedscgpt_smpc_table, out_path):
    """
    Plot two vertically stacked tables comparing FedscGPT and FedscGPT-SMPC,
    with zero cell padding, larger font, and a compact layout.
    """

    def _label_block(tbl, lbl):
        # prepend header
        header = ["Approach"] + tbl[0]
        rows = []
        for i, row in enumerate(tbl[1:]):
            # only the first data row gets the label; the rest are blank
            prefix = lbl if i == 1 else ""
            rows.append([prefix] + row)
        return [header] + rows

    block1 = _label_block(fedscgpt_table,"FedscGPT")
    block2 = _label_block(fedscgpt_smpc_table, "FedscGPT-SMPC")
    combined = block1 + block2[1:]  # skip second header
    n_cols = len(combined[0])
    col_widths = []
    for c in range(n_cols):
        max_len = max(len(str(r[c])) for r in combined)
        col_widths.append(max_len * 0.06)

    fig, ax = plt.subplots(figsize=(sum(col_widths) + 1, len(combined) * 0.4))
    ax.axis('off')
    tbl = ax.table(cellText=combined, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    for (row, col), cell in tbl.get_celld().items():
        cell.PAD = 0
        cell.set_width(col_widths[col] + 0.1)
        cell.set_height(0.3)
    tbl[(1, 0)].visible_edges = 'TLR'
    tbl[(2, 0)].visible_edges = 'LR'
    tbl[(3, 0)].visible_edges = 'LRB'
    tbl[(4, 0)].visible_edges = 'TLR'
    tbl[(5, 0)].visible_edges = 'LR'
    tbl[(6, 0)].visible_edges = 'LRB'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metric_cahnges_over_ER(file_path, epochs_list=[1, 2, 3, 4, 5], target_metric='Accuracy', img_format='svg'):
    df = pd.read_csv(file_path)
    df = df[df.Round < 11]
    df.Round = df.Round.astype(int)
    df.n_epochs = df.n_epochs.astype(int)
    datasets = df.Dataset.unique()

    # Set up the figure and subplots
    num_datasets = len(datasets)
    fig, axs = plt.subplots(1, num_datasets, figsize=(3 * num_datasets, 3), sharey=True)
    if num_datasets == 1:
        axs = [axs]  # Ensure axs is iterable if there's only one subplot

    # Set common x and y limits
    xlim = (df['Round'].min(), df['Round'].max())

    # Set font properties
    font_properties = {'fontsize': 12}

    # Iterate over datasets
    for i, dataset in enumerate(datasets):
        ax = axs[i]
        for n_epochs in epochs_list:
            # Filter data for the current dataset and number of epochs
            data = df[(df['Dataset'] == dataset) & (df['n_epochs'] == n_epochs) & (df['Metric'] == target_metric)]

            # Sort data by rounds for proper line plotting
            data = data.sort_values(by='Round')

            sns.lineplot(data=data, x='Round', y='Value', label=f'{n_epochs} Epochs', marker='o', ax=ax)

        # Set title and labels
        ax.set_title(handle_ds_name(dataset), **font_properties)
        ax.set_xlim(xlim)
        ax.set_ylim((0,1))

        if i == 0:
            ax.set_ylabel(target_metric, **font_properties)
        else:
            ax.set_ylabel('')


        ax.set_xlabel('Rounds', **font_properties)
        ax.set_xticks(list(range(1, int(max(ax.get_xticks())) + 1, 1)))



        ax.grid(True, which='both')
        ax.legend().set_visible(False)  # Hide individual legends

    # Add one common legend outside of the subplots
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.94), ncol=len(epochs_list), fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.8, wspace=0.051)  # Adjust top for the legend
    plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{target_metric}_changes_over_ER.{img_format}", format=img_format, dpi=300)


def find_best_fed(file_path, metric):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df = df[df['Metric'] == metric.title()]
    dataset_keys = df['Dataset'].unique()
    best = {}
    for dataset in dataset_keys:
        data = df[df['Dataset'] == dataset]
        best_metric = data['Value'].max()
        best_epoch = data.loc[data['Value'].idxmax(), 'n_epochs']
        best_round = data.loc[data['Value'].idxmax(), 'Round']
        print(f"For {dataset} the best {metric} is {best_metric} in {best_epoch} epochs and {best_round} rounds")
        best[dataset] = best_metric
    return best


def handle_image_format(start=True, img_frmt=None):
    global image_format
    if img_frmt:
        image_format = format
    if image_format == "svg":
        if start:
            plt.rcParams['svg.fonttype'] = 'none'
        else:
            plt.rcParams['svg.fonttype'] = 'path'


def handle_ds_name(name):
    if name.lower() == "hp":
        return "HP"
    if name.lower() == "ms":
        return "MS"
    if name.lower() == "myeloid":
        return "Myeloid"
    if name.lower() == "covid":
        return "Covid-19"
    if name.lower() == "lung":
        return "Lung-Kim"

class ImagePlaceholder:
    pass


def load_results_pkl(root_dir, pkl_file, best_fed):
    with open(pkl_file, 'rb') as f:
        result = pickle.load(f)
    results = {}
    for ds, res in best_fed.items():
        epochs = res['n_epochs']
        rounds = res['Round']
        results[ds] = {"federated": result[ds][epochs][rounds]}
        id_maps = result[ds]['id_maps']
        results[ds]['federated']['id_maps'] = id_maps
        results[ds]['federated']['unique_celltypes'] = list(id_maps.values())
        cent_pkl_file = os.path.join(root_dir, "output", "annotation", ds, 'centralized', 'results.pkl')
        if os.path.exists(cent_pkl_file):
            with open(cent_pkl_file, 'rb') as file:
                cent_res = pickle.load(file)
        results[ds]['centralized'] = cent_res
        results[ds]['centralized']['unique_celltypes'] = list(cent_res['id_maps'].values())
    return results
    # datasets = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    # results = {}
    #
    # for dataset in datasets:
    #     results[dataset] = {}
    #     for mode in ['centralized', 'federated']:
    #         pkl_file = os.path.join(root_dir, dataset, mode, 'results.pkl')
    #         results[dataset][mode] = {}
    #
    #         if os.path.exists(pkl_file):
    #             with open(pkl_file, 'rb') as file:
    #                 res = pickle.load(file)
    #                 results[dataset][mode]['metrics'] = res['results']
    #                 results[dataset][mode]['predictions'] = res['predictions']
    #                 results[dataset][mode]['labels'] = res['labels']
    #                 results[dataset][mode]['id_maps'] = res['id_maps']
    #                 results[dataset][mode]['unique_celltypes'] = list(res['id_maps'].values())
    #         else:
    #             print(f"results.pkl not found in {dataset} {mode}")
    # return results

def load_query_datasets(data_dir, dataset):
    candidates = [q for q in os.listdir(os.path.join(data_dir, dataset)) if q.startswith('query')]
    assert len(candidates) == 1, f"There are {len(candidates)} candidates for {dataset}"
    return anndata.read_h5ad(os.path.join(data_dir, dataset, candidates[0]))







def plot_confusion_matrices(dataset, results, color_mapping):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    modes = ['centralized', 'federated']
    annot_text_fontsize = {"hp": 20, "ms": 14, "myeloid": 20}
    for ax, mode in zip(axes, modes):
        predictions = results[dataset][mode]['predictions']
        labels = results[dataset][mode]['labels']
        id_maps = results[dataset][mode]['id_maps']
        unique_celltypes = results[dataset][mode]['unique_celltypes']
        color_mapping = {k: color_mapping[k] for k in unique_celltypes}



        # Filter celltypes
        for i in set([id_maps[p] for p in predictions]):
            if i not in unique_celltypes:
                unique_celltypes.remove(i)
        # Compute confusion matrix
        cm = confusion_matrix(labels, predictions)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_df = pd.DataFrame(cm, index=unique_celltypes[:cm.shape[0]], columns=unique_celltypes[:cm.shape[1]])
        cm_df = cm_df.round(2)
        # print(unique_celltypes)
        # print(color_mapping.keys())
        # print(cm_df)
        # exit()

        # Remove rows and columns with all NaNs
        nan_rows = [ind for i, ind in enumerate(cm_df.index) if all(cm_df.iloc[i].isna())]
        cm_df.drop(index=nan_rows, inplace=True)
        annot = cm_df.round(2).astype(str)
        annot[cm_df == 0] = '0'
        sns.heatmap(cm_df, annot=annot, fmt="", cmap="Blues", ax=ax, cbar=False, annot_kws={"size": annot_text_fontsize[dataset]})

        # Remove existing tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        color_loc = -0.02
        # Draw colored rectangles for y-tick labels (true labels)
        for ytick, cell_type in zip(ax.get_yticks(), cm_df.index):
            color = color_mapping.get(cell_type, 'black')
            rect = plt.Rectangle((color_loc, ytick - 0.5), 0.02, 1, color=color, transform=ax.get_yaxis_transform(),
                                 clip_on=False)
            ax.add_patch(rect)

        # Draw colored rectangles for x-tick labels (predicted labels)
        for xtick, cell_type in zip(ax.get_xticks(), cm_df.columns):
            color = color_mapping.get(cell_type, 'black')
            rect = plt.Rectangle((xtick - 0.5, 1), 1, 0.02, color=color,
                                 transform=ax.get_xaxis_transform(), clip_on=False)
            ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the position and size as needed
    plt.colorbar(axes[1].collections[0], cax=cbar_ax).ax.tick_params(labelsize=20)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{dataset}_confusion_matrices.svg", dpi=300, format='svg')
    plt.close()
    return color_mapping

def plot_umap_and_conf_matrix(root_dir, data_dir, res_pkl_file, res_df_file):
    df = pd.read_csv(res_df_file)
    best_fed = {ds: df.loc[df[(df.Dataset==ds) & (df.Metric == 'Accuracy')]["Value"].idxmax()] for ds in df.Dataset.unique()}
    results = load_results_pkl(root_dir, res_pkl_file, best_fed)

    for dataset in results.keys():
        adata = load_query_datasets(data_dir, dataset)
        predictions_centralized = results[dataset]['centralized']['predictions']
        predictions_federated = results[dataset]['federated']['predictions']
        labels = results[dataset]['centralized']['labels']  # Assuming the labels are the same for all modes
        unique_celltypes = results[dataset]['centralized']['unique_celltypes']
        assert results[dataset]['federated']['id_maps'] == results[dataset]['centralized']['id_maps']
        id_maps = results[dataset]['federated']['id_maps']
        labels = [id_maps[c] for c in labels]
        predictions_federated = [id_maps[c] for c in predictions_federated]
        predictions_centralized = [id_maps[c] for c in predictions_centralized]
        palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        palette_ = palette_ * 3  # Extend the palette if needed
        color_mapping = {c: palette_[i] for i, c in enumerate(unique_celltypes)}
        color_mapping = plot_confusion_matrices(dataset, results, color_mapping)

        plot_umaps(adata, predictions_centralized, predictions_federated, labels, unique_celltypes,
                   f"{dataset}_umap_plots.png", f"{dataset}_legend.png", color_mapping)
        plot_umaps(adata, predictions_centralized, predictions_federated, labels, unique_celltypes,
                   f"{dataset}_umap_plots.png", f"{dataset}_legend.png", color_mapping, plot_legend=True)


def plot_umaps(adata, predictions_centralized, predictions_federated, labels, unique_celltypes, file_name, legend_file_name, color_mapping, plot_legend=False):
    if 'X_umap' not in adata.obsm.keys():
        print(f"X_umap not found in adata.obsm {file_name} ==> the keys are: {adata.obsm.keys()}")
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
        sc.tl.umap(adata)
    adata.obs['cell_type'] = labels

    if not plot_legend:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (key, title) in zip(axes, [('cell_type', 'Annotated'), ('centralized', 'Centralized'), ('federated', 'Federated')]):
            if key != 'cell_type':
                adata.obs[key] = predictions_centralized if key == 'centralized' else predictions_federated
            sc.pl.umap(adata, color=key, palette=color_mapping, ax=ax, show=False, legend_loc=None)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{file_name}", dpi=300)
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(15, 5))
        sc.pl.umap(adata, color='cell_type', palette=color_mapping, show=False)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.close()
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: unique_celltypes.index(x[1]))
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)

        fig_legend, ax_legend = plt.subplots(figsize=(3, 9))  # Separate figure for the legend
        ax_legend.legend(sorted_handles, sorted_labels, loc='center', fontsize='small', frameon=False, ncol=1)
        ax_legend.axis('off')
        plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{legend_file_name}", dpi=300)
        plt.close(fig_legend)
        plt.close()


def create_metrics_dataframe(root_dir, res_df_file):
    # FedscGPT without SMPC
    fedscgpt = pd.read_csv(res_df_file)
    fedscgpt.dropna(inplace=True)
    fedscgpt_smpc = pd.read_csv(res_df_file.replace('.csv', '-smpc.csv'))
    fedscgpt_smpc.dropna(inplace=True)
    assert fedscgpt.shape[0] == fedscgpt_smpc.shape[0], "FedscGPT and FedscGPT-SMPC have different number of rows"
    results = {}
    for ds in fedscgpt.Dataset.unique():
        results[ds] = {"FedscGPT": find_best_performance(ds, fedscgpt),
                       "FedscGPT-SMPC": find_best_performance(ds, fedscgpt_smpc),
                       "scGPT": get_cent_performance(ds, os.path.join(root_dir, ds, 'centralized', 'results.pkl'))}

    rows = []

    for dataset, approaches in results.items():
        for approach, metrics in approaches.items():
            for metric, value in metrics.items():
                rows.append({
                    'Dataset': dataset,
                    'Approach': approach,
                    'Metric': metric,
                    'Value': value[0] if approach in ['FedscGPT', 'FedscGPT-SMPC'] else value,
                    'n_epochs': value[1] if approach in ['FedscGPT', 'FedscGPT-SMPC'] else None,
                    'Round': value[2] if approach in ['FedscGPT', 'FedscGPT-SMPC'] else None,
                })
    # Creating the DataFrame
    df = pd.DataFrame(rows)
    df.Metric = df.Metric.apply(lambda x: x[5:].title() if x.startswith('test/') else x)
    df.Metric = df.Metric.apply(lambda x: x[:-2] + "F1" if x.endswith('f1') else x)
    return df


def get_cent_performance(ds, cent_pkl_file):
    if os.path.exists(cent_pkl_file):
        with open(cent_pkl_file, 'rb') as file:
            cent_res = pickle.load(file)
    return cent_res['results']


def find_best_performance(ds, df):
    results = {}
    for metric in df.Metric.unique():
        idmax = df[(df.Dataset == ds) & (df.Metric == metric)]["Value"].idxmax()
        results[metric] = df.loc[idmax, ["Value", "n_epochs", "Round"]].values
    return results


def handle_metrics(metric):
    metric = metric[5:]
    if metric == "macro_f1":
        return 'Macro F1'
    return metric.title()

def plot_best_metrics(root_dir, param_tuning_df, img_format='svg'):
    handle_image_format(img_format)
    df = create_metrics_dataframe(root_dir, param_tuning_df)
    best_metrics_report(df)

    # Plot metrics with subplots for each metric and different modes as curves
    metrics = df['Metric'].unique()
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(4.5 * num_metrics, 5))


    for i, metric in enumerate(metrics):
        ax = axes[i] if num_metrics > 1 else axes
        metric_df = df[df['Metric'] == metric]
        sns.barplot(data=metric_df, x='Dataset', y='Value', hue='Approach', ax=ax, width=0.6)
        ax.set_ylabel(metric, fontsize=16)
        ax.set_ylim(0.4, 1)
        ax.tick_params(axis='both', which='major', labelsize=14)
        dataset_names = df['Dataset'].unique()
        ax.set_xticklabels([handle_ds_name(ds) for ds in dataset_names], fontsize=16)
        annotate_bars(ax, metric_df[metric_df['Approach'].isin(['FedscGPT', 'FedscGPT-SMPC'])])
    # Get the handles and labels from the last axis
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.05, 0.98), fontsize=16, ncol=3)

    # Remove legends from all axes
    for ax in axes:
        ax.get_legend().remove()
    legend_offset = 0.12
    plt.tight_layout(rect=[0, 0, 1 - legend_offset, 1 -legend_offset])  # Adjust rect to make space for the legend
    plt.savefig(f"{ANNOTATION_PLOTS_DIR}/best_metrics.{img_format}", dpi=300, format=img_format)
    plt.close()
    handle_image_format(start=False)


def annotate_bars(ax, df):
    """
    Annotate only FedscGPT / FedscGPT-SMPC bars by matching bar heights
    and dataset index (on the x-axis).
    """
    # The datasets in plotting order
    datasets = list(df['Dataset'].unique())
    # Only annotate these approaches
    targets = {'FedscGPT', 'FedscGPT-SMPC'}

    # Width used in seaborn.barplot
    bar_width = 0.6

    # Tolerance for matching heights
    tol = 1e-6

    for _, row in df.iterrows():
        ds = row['Dataset']
        approach = row['Approach']
        if approach not in targets:
            continue

        val = float(row['Value'])
        ds_idx = datasets.index(ds)

        # Look for the matching patch
        for p in ax.patches:
            # center of this bar
            x_center = p.get_x() + p.get_width() / 2
            height = p.get_height()

            # check dataset position and height match
            if (abs(x_center - ds_idx) < bar_width/2 + 0.01
                and abs(height - val) < tol):
                # annotate at 90% of bar height
                y_text = height * 0.9
                ep = int(row.get('n_epochs', row.get('epoch', 0)))
                nr = int(row.get('n_rounds', row.get('Round', 0)))
                print(f"Annotating {approach} for {ds} with Epoch: {ep}, Round: {nr}")
                ax.text(
                    x_center, y_text -0.05,
                    f"E:{ep}, R: {nr}",
                    ha='center', va='center',
                    rotation=90,
                    color='white',
                    fontsize=11,
                    zorder=10,
                    clip_on=False
                )
                break




def best_metrics_report(df):
    # Calculate the differences between federated and centralized
    df_pivot = df.pivot_table(index=['Dataset', 'Metric'], columns='Approach', values='Value').reset_index()
    df_pivot['Difference'] = df_pivot['FedscGPT-SMPC'] - df_pivot['scGPT']
    # Calculate the percentage of centralized performance achieved by federated learning
    df_pivot['Percentage Achieved'] = (df_pivot['FedscGPT-SMPC'] / df_pivot['scGPT']) * 100
    # Print the difference and percentage for each metric
    print("Differences and Percentage Achieved between FedscGPT-SMPC and scGPT for each metric:")
    print(df_pivot[['Dataset', 'Metric', 'Difference', 'Percentage Achieved']])
    # Identify and print the maximum difference
    max_diff_row = df_pivot.loc[df_pivot['Difference'].abs().idxmax()]
    print("\nMaximum Difference:")
    print(
        f"Dataset: {max_diff_row['Dataset']}, Metric: {max_diff_row['Metric']}, Difference: {max_diff_row['Difference']}")
    # Identify and print the worst case (lowest percentage achieved)
    worst_case_row = df_pivot.loc[df_pivot['Percentage Achieved'].idxmin()]
    print("\nWorst Case (Lowest Percentage Achieved):")
    print(
        f"Dataset: {worst_case_row['Dataset']}, Metric: {worst_case_row['Metric']}, Percentage Achieved: {worst_case_row['Percentage Achieved']}%")


def embedding_boxplot(data_dir, img_format='svg'):
    dataset = ["covid", "lung"]
    metrics = ['accuracy', 'precision', 'recall', 'macro_f1']

    # Initialize an empty DataFrame to hold all results
    df = pd.DataFrame(columns=['Dataset', 'Type', 'Metric', 'Value'])

    fedscgpt_file_path = {ds: f"{data_dir}/{ds}/federated/evaluation_metrics.csv" for ds in dataset}
    fedscgpt_smpc_file_path = {ds: f"{data_dir}/{ds}/federated/smpc/evaluation_metrics.csv" for ds in dataset}
    scgpt_file_path = {ds: f"{data_dir}/{ds}/centralized/evaluation_metrics.csv" for ds in dataset}

    for ds in dataset:
        # Load centralized and federated results
        scgpt = pd.read_csv(scgpt_file_path[ds])
        fedscgpt = pd.read_csv(fedscgpt_file_path[ds])
        fedscgpt_smpc = pd.read_csv(fedscgpt_smpc_file_path[ds])
        rows = []
        # Append centralized and federated results to the DataFrame
        for metric in metrics:
            rows.append({
                'Dataset': ds,
                'Type': 'scGPT',
                'Metric': metric,
                'Value': scgpt[metric].values[0]
            })
            rows.append({
                'Dataset': ds,
                'Type': 'FedscGPT',
                'Metric': metric,
                'Value': fedscgpt[metric].values[0]
            })
            rows.append({
                'Dataset': ds,
                'Type': 'FedscGPT-SMPC',
                'Metric': metric,
                'Value': fedscgpt_smpc[metric].values[0]
            })
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        # Collect and append client results
        client_dir_path = os.path.join(data_dir, ds, "centralized")
        rows = []
        for client_dir in os.listdir(client_dir_path):
            if client_dir.startswith("client"):
                client_metrics = pd.read_csv(os.path.join(client_dir_path, client_dir, "evaluation_metrics.csv"))
                for metric in metrics:
                    rows.append({
                        'Dataset': ds,
                        'Type': 'Client',
                        'Metric': metric,
                        'Value': client_metrics[metric].values[0]
                    })
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    # display_federated_performance_report(df)
    per_metric_annotated_scatterplot(df, "./plots/embedding", img_format)

def find_federated_performance_comparison(df, federated_types=None):
    """
    For each federated type in federated_types (default: FedscGPT & FedscGPT-SMPC),
    compare against centralized scGPT: compute Difference, Percentage, worst case,
    and categorize higher/equal/lower.
    Returns:
        results: { federated_type: (worst_report, higher_list, equal_list, lower_list, N) }
    """
    if federated_types is None:
        federated_types = ['FedscGPT', 'FedscGPT-SMPC']
    results = {}

    for fed in federated_types:
        perf = (
            df[df['Type'].isin(['scGPT', fed])]
            .pivot_table(index=['Dataset','Metric'], columns='Type', values='Value')
            .reset_index()
        )
        perf['scGPT'] = perf['scGPT'].round(3)
        perf[fed]   = perf[fed].round(3)
        perf['Difference'] = (perf[fed] - perf['scGPT']).round(3)
        perf['Percentage'] = (perf[fed] / perf['scGPT'] * 100).round(2)

        N = perf['Percentage'].min()
        worst = perf.loc[perf['Difference'].idxmin()]
        worst_report = (
            f"Worst performance of {fed} vs scGPT is on '{worst['Dataset']}' "
            f"for metric '{worst['Metric']}', Difference={worst['Difference']:.3f} "
            f"({fed}={worst[fed]:.3f}, scGPT={worst['scGPT']:.3f})."
        )

        higher = perf[perf['Difference'] > 0][['Dataset','Metric',fed,'scGPT']].values.tolist()
        equal  = perf[perf['Difference'] == 0][['Dataset','Metric',fed,'scGPT']].values.tolist()
        lower  = perf[perf['Difference'] < 0][['Dataset','Metric',fed,'scGPT']].values.tolist()

        results[fed] = (worst_report, higher, equal, lower, N)

    return results


def display_federated_performance_report(df):
    """
    Print side-by-side comparisons of FedscGPT vs scGPT and FedscGPT-SMPC vs scGPT.
    """
    reports = find_federated_performance_comparison(df)
    for fed, (worst_report, higher, equal, lower, N) in reports.items():
        print(f"=== Performance Comparison: {fed} vs scGPT ===")
        print(worst_report)
        print(f"\n{fed} consistently reached at least {N}% of scGPT's performance across all metrics and datasets.\n")

        if higher:
            print(f"{fed} outperformed scGPT in:")
            for ds, metric, p_fed, p_cen in higher:
                print(f" - {ds} | {metric}: {fed}={p_fed:.3f} > scGPT={p_cen:.3f}")
        if equal:
            print(f"\n{fed} matched scGPT in:")
            for ds, metric, p_fed, p_cen in equal:
                print(f" - {ds} | {metric}: {fed}={p_fed:.3f} = scGPT={p_cen:.3f}")
        if lower:
            print(f"\n{fed} trailed scGPT in:")
            for ds, metric, p_fed, p_cen in lower:
                print(f" - {ds} | {metric}: {fed}={p_fed:.3f} < scGPT={p_cen:.3f}")
        print("\n")


def get_clients_batch_value(h5ad_file_dir, ds_name):
    client_batch = anndata.read_h5ad(h5ad_file_dir).obs[
        get_batch_key(ds_name)].unique()
    client_batch = client_batch[0] if len(client_batch) == 1 else client_batch
    return client_batch


def get_batch_key(ds_name):
    if ds_name == "covid":
        return "str_batch"
    if ds_name == "lung":
        return "sample"
    if ds_name == "hp":
        return "batch"
    if ds_name == "ms":
        return "Factor Value[sampling site]"
    if ds_name == "myeloid":
        return "top4+rest"
    raise ValueError(f"Invalid dataset name: {ds_name}")

def get_cell_key(ds_name):
    if ds_name == "ms":
        return "Factor Value[inferred cell type - authors labels]"
    if ds_name == "myeloid":
        return "combined_celltypes"
    if ds_name == "hp":
        return "Celltype"
    if ds_name == "covid":
        return "celltype"
    if ds_name == "lung":
        return "cell_type"
    raise ValueError(f"Invalid dataset name: {ds_name}")


def get_reference_name(ds_name):
    if ds_name == "hp":
        return "reference_refined.h5ad"
    if ds_name == "myeloid":
        return "reference_adata.h5ad"
    return "reference.h5ad"

def shorten_batch_value(batch_value):
    replace = {
        'COVID-19 (query)': 'COVID-19',
        'Sun_sample4_TC': 'TC',
        'Sun_sample3_TB': 'TB',
        'HCL': 'HCL',
        '10X': '10X',
        'Oetjen_A': 'Oetjen',
        'Sanger_Meyer_2019Madissoon': 'Sanger',
        'Krasnow_distal 1a': 'Kras1a',
        'Krasnow_distal 2': 'Kras2',
        'Sun_sample1_CS': 'CS',
        'Sun_sample2_KC': 'KC',
        'Krasnow_distal 3': 'Kras3',
        'P0034': 'P34',
        'P0028': 'P28',
        'P0025': 'P25',
        'P1028': 'P1028',
        'P0006': 'P6',
        'P0020': 'P20',
        'P0008': 'P8',
        'P0018': 'P18',
        'P0030': 'P30',
        'P1058': 'P1058',
        'client_cerebral cortex': "Cerebral",
        'client_prefrontal cortex': 'Prefrontal',
        'client_premotor cortex': 'Premotor'
    }
    return replace.get(batch_value, batch_value)

def per_metric_annotated_scatterplot(df, plots_dir, img_format='svg', proximity_threshold=0.1):
    """
    Plot annotated scatterplots per metric, including:
      - Per-client points (jittered + batch labels)
      - scGPT    as short horizontal dashed lines
      - FedscGPT as diamond markers
      - FedscGPT-SMPC as star markers
      - Any other 'Type' as fallback X-markers

    Parameters:
    - df: DataFrame with columns ['Dataset','Type','Metric','Value','Batch']
    - plots_dir: output directory for plots
    - img_format: image format (e.g. 'svg')
    - proximity_threshold: how close y-values must be to trigger label-offset logic
    """

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    datasets = df['Dataset'].unique()
    metrics  = df['Metric'].unique()

    # Identify client vs. other approach rows
    client_mask = df['Type'].str.contains('client', case=False)
    client_types = df.loc[client_mask, 'Type'].unique()
    other_types  = [t for t in df['Type'].unique() if t not in client_types]

    # Style map for known approaches; fallback to X
    style_map = {
        'scGPT':           {'kind':'line', 'linestyle':'--', 'linewidth':2},
        'FedscGPT':        {'marker':FEDSCGPT_MARKER, 's':100, 'edgecolor':'black'},
        'FedscGPT-SMPC':   {'marker':FEDSCGPT_SMPC_MARKER, 's':100, 'edgecolor':'black'}
    }
    default_marker = 'X'

    for metric in metrics:
        plt.figure(figsize=(5,5))
        # --- Plot client points with jitter + labels ---
        cmap = plt.get_cmap('tab10').colors
        for i, ds in enumerate(datasets):
            csub = df[(df['Metric']==metric)&(df['Dataset']==ds)&client_mask]
            if csub.empty:
                continue
            vals    = csub['Value'].values
            batches = csub.get('Batch', ['']*len(vals))
            # jitter
            jitter = 0.05
            x_base = i+1
            x_j    = x_base + np.random.uniform(-jitter, jitter, size=len(vals))
            plt.scatter(x_j, vals,
                        color=cmap[i%len(cmap)],
                        edgecolor='black', s=50, alpha=0.7)

            # annotate with batch, offsetting if many neighbors
            for j, (x,y,b) in enumerate(zip(x_j, vals, batches)):
                label = shorten_batch_value(b)
                close_count = np.sum(np.abs(vals - y) < proximity_threshold)
                if close_count>1 and (j%2)==1:
                    ha, x_off = 'right', -0.05
                else:
                    ha, x_off = 'left', 0.05
                plt.text(x + x_off, y, label, fontsize=10, ha=ha, va='center')

        # --- Overlay each federated/centralized approach ---
        for typ in other_types:
            for i, ds in enumerate(datasets):
                sub = df[(df['Metric']==metric)&(df['Dataset']==ds)&(df['Type']==typ)]
                if sub.empty:
                    continue
                val = sub['Value'].values[0]
                style = style_map.get(typ, {})
                x = i+1
                color = cmap[i % len(cmap)],

                if style.get('kind')=='line':
                    # short horizontal line
                    plt.hlines(val, x-0.3, x+0.3,
                               linestyle=style['linestyle'],
                               linewidth=style['linewidth'],
                               color=color,
                               zorder=3)
                else:
                    m      = style.get('marker', default_marker)
                    size   = style.get('s', 80)
                    ec     = style.get('edgecolor','black')
                    plt.scatter(x, val,
                                marker=m,
                                s=size,
                                edgecolor=ec,
                                zorder=5)

        # --- Final tweaks & save ---
        plt.xlabel('')
        plt.ylim(0, 1)
        plt.ylabel(metric.capitalize(), fontsize=18)
        plt.xticks(range(1, len(datasets)+1),
                   [handle_ds_name(d) for d in datasets],
                   fontsize=18)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"{metric}_scatterplot_annotated.{img_format}")
        plt.savefig(out_path, dpi=300, format=img_format, bbox_inches='tight')
        plt.close()

        # regenerate legend (assumed your helper handles all markers/lines)
        plot_legend(plots_dir, img_format)

def plot_legend(plots_dir, img_format='svg'):
    """
    Plot a separate figure containing only the legend for:
      – scGPT (centralized)
      – FedscGPT (federated)
      – FedscGPT-SMPC (federated + SMPC)
      – Clients
      – Other approaches
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.figure(figsize=(6, 1.5))

    legend_elements = [
        Line2D([0], [0],
               color='black', lw=2, linestyle='--',
               label='scGPT'),
        Line2D([0], [0],
               marker=FEDSCGPT_MARKER, color='w', markersize=10,
               markeredgecolor='black',
               label='FedscGPT'),
        Line2D([0], [0],
               marker=FEDSCGPT_SMPC_MARKER, color='w', markersize=10,
               markeredgecolor='black',
               label='FedscGPT-SMPC'),
        Line2D([0], [0],
               marker='o', color='w', markersize=8,
               markeredgecolor='black',
               label='Clients'),
    ]

    plt.legend(handles=legend_elements,
               loc='center', fontsize=12,
               ncol=len(legend_elements),
               frameon=False,
               columnspacing=1.0)

    plt.axis('off')
    plt.savefig(f"{plots_dir}/legend.{img_format}",
                format=img_format, dpi=300,
                bbox_inches='tight')
    plt.close()

def plot_embedding_boxplot(df, img_format='svg'):
    """
    Plot data using Matplotlib from a pandas DataFrame.
    """
    if not os.path.exists('./plots/embedding'):
        os.makedirs('./plots/embedding')
    metrics = df['Metric'].unique()
    datasets = df['Dataset'].unique()

    for metric in metrics:
        plt.figure(figsize=(5, 5))

        # Separate client data and centralized/federated data
        client_data = df[(df['Metric'] == metric) & (df['Type'].str.contains('Client'))]
        scgpt = df[(df['Metric'] == metric) & (df['Type'] == 'scGPT')]
        fedscgpt = df[(df['Metric'] == metric) & (df['Type'] == 'FedscGPT')]
        fedscgpt_smpc = df[(df['Metric'] == metric) & (df['Type'] == 'FedscGPT-SMPC')]

        # Prepare data for boxplot
        client_values = [client_data[client_data['Dataset'] == dataset]['Value'].values for dataset in datasets]

        # Create boxplots
        box = plt.boxplot(client_values, patch_artist=True, positions=range(1, len(datasets) + 1))

        # Colors for boxplots
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgrey', 'lightyellow']  # Extend as needed
        for patch, color in zip(box['boxes'], colors[:len(box['boxes'])]):
            patch.set_facecolor(color)

        # Overlay centralized and federated data points
        for i, dataset in enumerate(datasets):
            # Centralized as dashed lines
            plt.axhline(y=scgpt[scgpt['Dataset'] == dataset]['Value'].values[0],
                        color=box['boxes'][i].get_facecolor(), linestyle='--', linewidth=2, zorder=3)
            # FedscGPT as scatter points
            plt.scatter(i + 1, fedscgpt[fedscgpt['Dataset'] == dataset]['Value'].values[0],
                        color=box['boxes'][i].get_facecolor(), edgecolor='black', zorder=5, marker=FEDSCGPT_MARKER, s=100)
            # FedscGPT-SMPC as scatter points
            plt.scatter(i + 1, fedscgpt_smpc[fedscgpt_smpc['Dataset'] == dataset]['Value'].values[0],
                        color=box['boxes'][i].get_facecolor(), edgecolor='black', zorder=5, marker=FEDSCGPT_SMPC_MARKER, s=100)

        # Customize the plot
        plt.xlabel('Datasets', fontsize=16)
        plt.ylabel(metric.capitalize(), fontsize=16)
        plt.xticks(range(1, len(datasets) + 1), [handle_ds_name(d) for d in datasets], fontsize=16)
        plt.yticks(fontsize=16)


        boxplot_image_path = 'boxplot.png'
        boxplot_image_pil = Image.open(boxplot_image_path).convert("RGBA")  # Ensure it is RGBA
        # Convert to an array suitable for Matplotlib
        boxplot_image = np.array(boxplot_image_pil)
        image_placeholder_instance = ImagePlaceholder()
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, linestyle='--', label='scGPT'),
            Line2D([0], [0], marker=FEDSCGPT_MARKER, color='w', markersize=10, label='FedscGPT',
                   markeredgecolor='black'),
            Line2D([0], [0], marker=FEDSCGPT_SMPC_MARKER, color='w', markersize=10, label='FedscGPT-SMPC',
                     markeredgecolor='black'),
            image_placeholder_instance
        ]
        legend_labels = ['scGPT', 'FedscGPT', 'FedscGPT-SMPC', 'Clients']

        class HandlerImage:
            def __init__(self, image):
                self.image = image

            def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                x0, y0 = handlebox.xdescent, handlebox.ydescent
                imagebox = OffsetImage(self.image, zoom=0.15)
                ab = AnnotationBbox(imagebox, (x0 + .61, y0 + .08), frameon=False, xycoords='axes fraction')
                handlebox.add_artist(ab)
                return ab

        plt.legend(handles=legend_elements, labels=legend_labels, loc='lower right', fontsize=14, labelspacing=0.9,
                   borderpad=.3, handlelength=1.4,
                   handler_map={image_placeholder_instance: HandlerImage(boxplot_image)})

        plt.tight_layout()
        plt.savefig(f"./plots/embedding/{metric}_boxplot.{img_format}", format=img_format, dpi=300)
        plt.close()

def fed_embedding_umap(data_dir, res_dir, img_format='svg'):
    datasets = ["lung", "covid"]
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"] * 3

    for ds in datasets:
        print(f"Plotting UMAP for {ds}…")
        cache_fp = f"./plots/embedding/{ds}_umap.pickle"

        # 1) Load reference & query once
        ref_fp = os.path.join(data_dir, ds, get_reference_name(ds))
        qry_fp = os.path.join(data_dir, ds, "query.h5ad")
        reference = anndata.read_h5ad(ref_fp)
        query     = anndata.read_h5ad(qry_fp)
        query.obs["preds"] = pd.read_csv(
            os.path.join(res_dir, ds, "federated/smpc/preds.csv")
        )["predictions"].astype(str).values

        batch_key = get_batch_key(ds)
        cell_key  = get_cell_key(ds)

        if os.path.exists(cache_fp):
            # 2a) inject already‐computed UMAP & obs
            with open(cache_fp, "rb") as f:
                data = pickle.load(f)
            reference.obsm["X_umap"] = data["reference_umap"]
            query.obsm    ["X_umap"] = data["query_umap"]
            reference.obs = data["reference_obs"].copy()
            query.obs     = data["query_obs"].copy()
        else:
            # 2b) compute & cache
            reference.obs[batch_key] = reference.obs[batch_key].apply(shorten_batch_value).astype(str)
            reference.obs[cell_key]  = reference.obs[cell_key].astype(str)
            query.obs[cell_key]      = query.obs[cell_key].astype(str)

            for obj in (reference, query):
                if "X_umap" not in obj.obsm:
                    sc.pp.neighbors(obj, n_neighbors=30, use_rep="X")
                    sc.tl.umap(obj)

            os.makedirs(os.path.dirname(cache_fp), exist_ok=True)
            with open(cache_fp, "wb") as f:
                pickle.dump({
                    "reference_umap": reference.obsm["X_umap"],
                    "query_umap":     query.obsm["X_umap"],
                    "reference_obs":  reference.obs[[batch_key, cell_key]],
                    "query_obs":      query.obs[[cell_key, "preds"]],
                }, f)

        # 3) build color maps from the (possibly replaced) obs
        unique_celltypes = sorted(reference.obs[cell_key].unique())
        unique_batches   = sorted(reference.obs[batch_key].unique())
        if len(unique_celltypes) > len(palette):
            palette *= (len(unique_celltypes) // len(palette) + 1)

        cell_color_mapping  = {c: colors.to_hex(palette[i]) for i, c in enumerate(unique_celltypes)}
        batch_color_mapping = {b: colors.to_hex(palette[i]) for i, b in enumerate(unique_batches)}

        # 4) plot the 2×2 UMAP panel
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        sc.pl.umap(reference, color=batch_key,
                   palette=batch_color_mapping,
                   ax=axes[0][0], show=False, legend_loc=None)
        axes[0][0].axis("off")

        sc.pl.umap(reference, color=cell_key,
                   palette=cell_color_mapping,
                   ax=axes[0][1], show=False, legend_loc=None)
        axes[0][1].axis("off")

        sc.pl.umap(query, color=cell_key,
                   palette=cell_color_mapping,
                   ax=axes[1][0], show=False, legend_loc=None)
        axes[1][0].axis("off")

        sc.pl.umap(query, color="preds",
                   palette=cell_color_mapping,
                   ax=axes[1][1], show=False, legend_loc=None)
        axes[1][1].axis("off")

        out_fp = f"./plots/embedding/umap-{ds}.{img_format}"
        plt.tight_layout()
        plt.savefig(out_fp, format=img_format, dpi=300)
        plt.close(fig)
        print(f"  → saved {out_fp}")
        plot_umap_legend(reference, cell_key, batch_key, cell_color_mapping, batch_color_mapping, unique_celltypes, unique_batches, ds, img_format)


def plot_umap_legend(
    reference,
    cell_key,
    batch_key,
    cmap_cells,
    cmap_batches,
    u_cells,
    u_batches,
    ds,
    img_format='svg'
):
    """
    reference: AnnData with UMAP already in .obsm['X_umap']
    cell_key/batch_key: obs column names
    cmap_cells/cmap_batches: dicts mapping labels -> hex colors
    u_cells/u_batches: lists of labels in desired legend order
    ds: dataset name (used for filenames)
    """

    # 1) Cell‐type legend
    fig = plt.subplots(figsize=(5, 5))
    sc.pl.umap(
        reference,
        color=cell_key,
        palette=cmap_cells,
        show=False,
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.close()
    sorted_ct = sorted(zip(handles, labels), key=lambda x: u_cells.index(x[1]))
    h_ct, l_ct = zip(*sorted_ct)

    ct_fig, ct_ax = plt.subplots(figsize=(3, len(u_cells) * 0.3))
    ct_ax.legend(
        h_ct,
        l_ct,
        loc='center',
        frameon=False,
        fontsize='small',
        ncol=1
    )
    ct_ax.axis('off')
    ct_fp = f"./plots/embedding/umap-celltype-legend-{ds}.{img_format}"
    plt.savefig(ct_fp, dpi=300, format=img_format, bbox_inches='tight')
    plt.close()

    # 2) Batch legend
    fig = plt.subplots(figsize=(5, 5))
    sc.pl.umap(
        reference,
        color=batch_key,
        palette=cmap_batches,
        show=False,
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.close()

    sorted_bt = sorted(zip(handles, labels), key=lambda x: u_batches.index(x[1]))
    h_bt, l_bt = zip(*sorted_bt)

    bt_fig, bt_ax = plt.subplots(figsize=(3, len(u_batches) * 0.3))
    bt_ax.legend(
        h_bt,
        l_bt,
        loc='center',
        frameon=False,
        fontsize='small',
        ncol=1
    )
    bt_ax.axis('off')
    bt_fp = f"./plots/embedding/umap-batch-legend-{ds}.{img_format}"
    plt.savefig(bt_fp, dpi=300, format=img_format, bbox_inches='tight')
    plt.close()
    print(f"  → saved batch legend {bt_fp}")

def accuracy_annotated_scatterplot(df, plots_dir, img_format='svg', proximity_threshold=0.2):
    """
    Plot data using Matplotlib from a pandas DataFrame, with each scatter point annotated by its corresponding 'Batch' value.
    Adjusts the text to the left or right dynamically to avoid overlap for points with close y-values.

    Parameters:
    - df: DataFrame containing the data to plot.
    - plots_dir: Directory to save the plots.
    - img_format: Format to save the images (e.g., 'svg').
    - proximity_threshold: Defines the closeness of y-values to consider them overlapping (default = 0.1).
    - legend_inside: Boolean flag indicating whether to place the legend inside the figure (default = False).
    """
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    metrics = df['Metric'].unique()
    datasets = df['Dataset'].unique()
    print(datasets)

    for metric in metrics:
        plt.figure(figsize=(5, 5))

        # Separate client data and centralized/federated data
        df_metric = df[df['Metric'] == metric]
        scgpt = df_metric[df_metric['Type'] == 'scGPT']
        fedscgpt_smpc = df_metric[df_metric['Type'] == 'FedscGPT-SMPC']
        fedscgpt = df_metric[df_metric['Type'] == 'FedscGPT']
        client_data = df_metric[~df_metric['Type'].isin(['scGPT', 'FedscGPT-SMPC', 'FedscGPT'])]

        # Scatter plot for client data
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgrey', 'lightyellow']  # Extend as needed
        scatter_plots = []
        for i, dataset in enumerate(datasets):
            dataset_clients = client_data[client_data['Dataset'] == dataset]
            client_values = dataset_clients['Value'].values
            client_batches = dataset_clients['Type'].values
            # Scatter each client point with a slight horizontal offset to avoid overlap
            jitter = 0.05  # Add some horizontal jitter to avoid overlap
            x_jitter = np.random.uniform(-jitter, jitter, size=client_values.shape)
            scatter = plt.scatter([i + 1 + x for x in x_jitter], client_values,
                                  color=colors[i % len(colors)], edgecolor='black', s=50, alpha=0.7,
                                  label=f"Client Data - {dataset}")
            scatter_plots.append(scatter)

            # Determine proximity of y-values to decide label positions
            for j, (x, y, batch) in enumerate(zip([i + 1 + x for x in x_jitter], client_values, client_batches)):
                batch_label = shorten_batch_value(batch)

                # Check if other points are "close enough" in y-value using the proximity threshold
                close_points = np.sum(np.abs(client_values - y) < proximity_threshold)
                annotation_font_size = 12
                if close_points > 1:  # If there are other points within the threshold range
                    # Alternate placement of labels for overlapping points
                    if j % 2 == 0:
                        plt.text(x + 0.1, y, batch_label, fontsize=annotation_font_size, ha='left', va='center')
                    else:
                        plt.text(x - 0.1, y, batch_label, fontsize=annotation_font_size, ha='right', va='center')
                else:
                    plt.text(x + 0.1, y, batch_label, fontsize=annotation_font_size, ha='left', va='center')

        # Overlay centralized and federated data points
        for i, dataset in enumerate(datasets):
            # Centralized as horizontal lines only within the dataset range
            if not scgpt[scgpt['Dataset'] == dataset].empty:
                scgpt_value = scgpt[scgpt['Dataset'] == dataset]['Value'].values[0]
                plt.hlines(y=scgpt_value, xmin=i + 0.7, xmax=i + 1.3,
                           color=colors[i % len(colors)], linestyle='--', linewidth=2, zorder=3,
                           label=f"scGPT")

            # Federated as scatter points
            if not fedscgpt[fedscgpt['Dataset'] == dataset].empty:
                federated_value = fedscgpt[fedscgpt['Dataset'] == dataset]['Value'].values[0]
                plt.scatter(i + 1, federated_value, color=colors[i % len(colors)], edgecolor='black',
                            zorder=5, marker=FEDSCGPT_MARKER, s=100, label=f"FedscGPT")

            if not fedscgpt_smpc[fedscgpt_smpc['Dataset'] == dataset].empty:
                federated_value = fedscgpt_smpc[fedscgpt_smpc['Dataset'] == dataset]['Value'].values[0]
                plt.scatter(i + 1, federated_value, color=colors[i % len(colors)], edgecolor='black',
                            zorder=5, marker=FEDSCGPT_SMPC_MARKER, s=100, label=f"FedscGPT-SMPC")



        # Customize the plot
        plt.xlabel('', fontsize=1)
        plt.ylabel(metric.capitalize(), fontsize=20)
        plt.ylim(0, 1)
        plt.xticks(range(1, len(datasets) + 1), [handle_ds_name(d) for d in datasets], fontsize=20)
        plt.yticks(fontsize=18)

        # Legend placement based on the flag
        custom_handles = [
            plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='white', markersize=8, linestyle='None',
                        label='Clients'),
            plt.Line2D([0], [0], marker=FEDSCGPT_MARKER, color='black', markerfacecolor='white', markersize=10, linestyle='None',
                        label='FedscGPT'),
            plt.Line2D([0], [0], marker=FEDSCGPT_SMPC_MARKER, color='black', markerfacecolor='white', markersize=10, linestyle='None',
                       label='FedscGPT-SMPC'),
            plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='scGPT')
        ]
        legend = plt.legend(handles=custom_handles, loc='lower left', fontsize=14, frameon=True)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_facecolor('white')

        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{metric}_scatterplot_annotated.{img_format}", format=img_format, dpi=300,
                    bbox_inches='tight')
        plt.close()


def plot_batch_effect_umaps(raw_h5ad, cent_corrected, fed_corrected, batch_key, cell_key, out_prefix, standalone=False):
    """
    Plot a 2×3 grid of UMAPs:
      Top row: Raw, Centralized, Federated colored by cell type.
      Bottom row: Raw, Centralized, Federated colored by batch.
    After computing UMAP (if missing), save it back to the original file.

    Parameters
    ----------
    raw_h5ad : str
        Path to raw (uncorrected) AnnData file.
    cent_corrected : str
        Path to centralized-corrected AnnData file.
    fed_corrected : str
        Path to federated-corrected AnnData file.
    batch_key : str
        Key in adata.obs for batch labels.
    cell_key : str
        Key in adata.obs for cell type labels.
    out_prefix : str
        Prefix for the saved plot file (without extension).
    """
    # Load AnnData objects
    names = ["Raw", "Centralized", "Federated"]
    paths = [raw_h5ad, cent_corrected, fed_corrected]
    adatas = {}
    for name, path in zip(names, paths):
        adata = sc.read_h5ad(path)
        # Compute UMAP if not present
        if "X_umap" not in adata.obsm:
            sc.pp.neighbors(adata, use_rep="X", n_neighbors=30)
            sc.tl.umap(adata)
            adata.write(path)  # overwrite file with UMAP stored
        adatas[name] = adata
    # drop standalone cell types
    if not standalone:
        stand_alone_celltypes = ["CCR7+ T", "CD8 T", "CD10+ B cells", "Ciliated", "DC_activated", "Erythrocytes",
                                "Erythroid progenitors", "IGSF21+ Dendritic", "M2 Macrophage", "Megakaryocytes",
                                "Monocyte progenitors", "Proliferating T", "Signaling Alveolar Epithelial Type 2",
                                "Treg", "Tregs", "innate T"]
        for name, adata in adatas.items():
            adatas[name] = adata[~adata.obs[cell_key].isin(stand_alone_celltypes)]
    uniq_ct = list(adatas["Centralized"].obs[cell_key].cat.categories)
    cell_palette = generate_palette(uniq_ct)
    # batches come from the raw data
    uniq_batch = list(adatas["Raw"].obs[batch_key].cat.categories)
    batch_palette = generate_palette(uniq_batch)

    # 3) Set up 2x3 grid + space for legends on right
    fig, axes = plt.subplots(2, 3, figsize=(18, 12),
                             gridspec_kw={"right": 0.75})

    # 4) Plotting
    for col, name in enumerate(names):
        ad = adatas[name]
        # Top: cell type
        ax_ct = axes[0, col]
        sc.pl.umap(ad, color=cell_key, ax=ax_ct, show=False,
                   palette=cell_palette, legend_loc=None)
        ax_ct.set_title(f"{name} (cell type)")
        ax_ct.axis("off")
        # Bottom: batch
        ax_bt = axes[1, col]
        sc.pl.umap(ad, color=batch_key, ax=ax_bt, show=False,
                   palette=batch_palette, legend_loc=None)
        ax_bt.set_title(f"{name} (batch)")
        ax_bt.axis("off")

    # 5) Create external legends
    # Cell‐type legend
    ct_handles = [
        plt.Line2D([0], [0], marker='o', color=cell_palette[ct], linestyle='', label=ct)
        for ct in uniq_ct
    ]
    # Batch legend
    bt_handles = [
        plt.Line2D([0], [0], marker='o', color=batch_palette[b], linestyle='', label=b)
        for b in uniq_batch
    ]

    # Place legends
    # Cell‐type on upper right
    fig.legend(handles=ct_handles,
               loc="upper right",
               bbox_to_anchor=(1, 0.98),
               title=cell_key,
               ncol=1, fontsize='small', title_fontsize='medium')
    # Batch on lower right
    fig.legend(handles=bt_handles,
               loc="lower right",
               bbox_to_anchor=(1, 0.05),
               title=batch_key,
               ncol=1, fontsize='small', title_fontsize='medium')
    fig.subplots_adjust(left=0.01, right=0.75)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=300)
    plt.close(fig)