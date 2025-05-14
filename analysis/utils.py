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

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

set_seed(SEED)
image_format = 'svg'
ANNOTATION_PLOTS_DIR = 'plots/annotation'

def load_metric(filepath, metric):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data['results'][f'test/{metric}']

def collect_metrics(base_path, metric):
    accuracies = {}
    for root, dirs, files in os.walk(base_path):
        if 'results.pkl' in files:
            client_name = os.path.basename(root)
            accuracy = load_metric(os.path.join(root, 'results.pkl'), metric)
            accuracies[client_name] = accuracy
    return accuracies


class CentralizedMetricPlotter:

    @staticmethod
    def collect_data(metrics):
        """
        Collect data from metrics into a pandas DataFrame.
        """
        data = []
        for dataset, values in metrics.items():
            client_acc = [acc for key, acc in values.items() if key not in ['centralized', 'federated']]
            centralized_acc = values['centralized']
            federated_acc = values['federated']

            # Append each client's data
            for acc in client_acc:
                data.append({'Dataset': dataset, 'Type': 'Client', 'Accuracy': acc})

            # Append centralized accuracy
            data.append({'Dataset': dataset, 'Type': 'Centralized', 'Accuracy': centralized_acc})

            # Append federated accuracy
            data.append({'Dataset': dataset, 'Type': 'Federated', 'Accuracy': federated_acc})

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

    @staticmethod
    def plot_data_matplotlib(df, metric_name, plot_name, img_format='svg'):
        """
        Plot data using Matplotlib from a pandas DataFrame.
        """
        datasets = df['Dataset'].unique()
        client_accuracies = [df[(df['Dataset'] == dataset) & (df['Type'] == 'Client')]['Accuracy'].values for dataset in
                             datasets]
        centralized_accuracies = df[df['Type'] == 'Centralized'].set_index('Dataset')['Accuracy']
        federated_accuracies = df[df['Type'] == 'Federated'].set_index('Dataset')['Accuracy']

        plt.figure(figsize=(5, 5))
        # Create boxplots
        box = plt.boxplot(client_accuracies, patch_artist=True, positions=range(1, len(datasets) + 1))
        # Colors for boxplots
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgrey', 'lightyellow']  # Extend as needed
        for patch, color in zip(box['boxes'], colors[:len(box['boxes'])]):
            patch.set_facecolor(color)

        # Add centralized accuracies as horizontal dashed lines
        for i, dataset in enumerate(datasets):
            plt.axhline(y=centralized_accuracies[dataset], color=box['boxes'][i].get_facecolor(), linestyle='--',
                        linewidth=2, zorder=3)
            plt.scatter(i + 1, federated_accuracies[dataset], color=box['boxes'][i].get_facecolor(), edgecolor='black',
                        zorder=5, marker='*', s=100)

        boxplot_image_path = 'boxplot.png'
        boxplot_image_pil = Image.open(boxplot_image_path).convert("RGBA")  # Ensure it is RGBA
        # Convert to an array suitable for Matplotlib
        boxplot_image = np.array(boxplot_image_pil)
        image_placeholder_instance = ImagePlaceholder()
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, linestyle='--', label='Centralized'),
            Line2D([0], [0], marker='o', color='w', markersize=10, label='outlier',
                   markeredgecolor='black'),
            Line2D([0], [0], marker='*', color='w', markersize=10, label='Federated',
                   markeredgecolor='black'),
            image_placeholder_instance
        ]
        legend_labels = ['Centralized', 'Outlier', 'Federated', 'Clients']

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
    df = df[~(df.Round == 0)]
    # Determine unique datasets and metrics
    dataset_keys = df['Dataset'].unique()
    metric_keys = df['Metric'].unique()

    fig, axs = plt.subplots(len(metric_keys), len(dataset_keys),
                            figsize=(len(dataset_keys) *7, len(df['n_epochs'].unique()) + 1),
                            squeeze=True)
    fig.subplots_adjust(left=0.08, right=0.8, top=0.9, bottom=0.1, wspace=0.01, hspace=0.01)

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




def analyze_communication_efficiency(results_file_path, centralized_file_path, percentages=[70, 80, 90, 92, 95, 99],
                                     metric="Accuracy"):
    """
    Analyze the communication efficiency by calculating the number of communication rounds and epochs required
    to reach specified percentages of the centralized accuracy and display the results in a table.

    Args:
        results_file_path (str): Path to the CSV file containing the federated results.
        centralized_file_path (str): Path to the CSV file containing the centralized results.
        percentages (list): List of percentages of centralized accuracy to target.

    Parameters
    ----------
    percentages
    centralized_file_path
    results_file_path
    metric
    """
    # Load the results DataFrame and centralized results
    df = pd.read_csv(results_file_path)
    df = df[~(df.Round == 0)]
    centralized_df = pd.read_csv(centralized_file_path)
    # Extract unique datasets and metrics
    dataset_keys = df['Dataset'].unique()

    # Store table data
    table_data = [["Dataset"] + [f"{p}%" for p in percentages]]

    # Iterate over each dataset
    for dataset in dataset_keys:
        row = [handle_ds_name(dataset)]
        for p in percentages:
            # Get the centralized accuracy for the specific dataset and metric
            central_value = centralized_df[(centralized_df['Dataset'] == dataset) & (centralized_df['Type'] == 'Centralized')][metric].values[0]

            # Filter the data for the current dataset and metric
            data = df[(df['Dataset'] == dataset) & (df['Metric'] == metric)]

            # Calculate the target value based on the centralized accuracy
            target_value = central_value * (p / 100)

            # Initialize variables to store the number of rounds and epochs for this percentage
            rounds_needed = None
            epochs_needed = None

            # Iterate over data sorted by rounds to find the first occurrence where the target value is met or exceeded
            max_rounds = data['Round'].max()
            for r in range(1, max_rounds + 1):
                rounds_data = data[data['Round'] == r]
                m = rounds_data['Value'].idxmax()
                if rounds_data.loc[m]['Value'] >= target_value:
                    rounds_needed = r
                    epochs_needed = rounds_data.loc[m]['n_epochs']
                    best_= rounds_data.loc[m]['Value']
                    break

            if rounds_needed is not None:
                row.append(f"{rounds_needed}|{epochs_needed}")
                print(f"For {dataset}, {p}% of centralized accuracy is reached in {rounds_needed} rounds and {epochs_needed} epochs with {best_}")
            else:
                row.append("NR")

        table_data.append(row)
    # Calculate column widths based on the longest text in each column
    col_widths = []
    for col_idx in range(len(table_data[0])):
        max_len = max(len(str(table_data[row_idx][col_idx])) for row_idx in range(len(table_data)))
        col_widths.append(max_len * 0.2)  # Adjust this multiplier as needed for text size

    # Plotting the table
    fig, ax = plt.subplots(figsize=(sum(col_widths) + 0.7, len(table_data) * 0.5))
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Set column widths based on calculated values
    for col_idx, width in enumerate(col_widths):
        table.auto_set_column_width(col=col_idx)  # Ensure the column auto width is set
        for row_idx in range(len(table_data)):
            table[(row_idx, col_idx)].set_width(width + 0.5)
    for key, cell in table.get_celld().items():
        cell.set_height(0.2)
    plt.savefig(f"{ANNOTATION_PLOTS_DIR}/communication.svg", dpi=300, format="svg")

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

def load_query_datasets(data_dir):
    datasets = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    query_datasets = {}
    for dataset in datasets:
        candidates = [q for q in os.listdir(os.path.join(data_dir, dataset)) if q.startswith('query')]
        if len(candidates) > 1:
            raise ValueError(f"There are {len(candidates)} candidates for {dataset}")
        query_datasets[dataset] = anndata.read_h5ad(os.path.join(data_dir, dataset, candidates[0]))
    return query_datasets






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
    query_datasets = load_query_datasets(data_dir)

    for dataset in results.keys():
        # plot_confusion_matrices(dataset, results)
        adata = query_datasets[dataset]
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
    fedscgpt_smpc = pd.read_csv(res_df_file.replace('.csv', '-smpc.csv'))
    results = {}
    for ds in fedscgpt.Dataset.unique():
        results[ds] = {"FedscGPT": find_best_performance(ds, fedscgpt),
                       "FedscGPT-SMPC": find_best_performance(ds, fedscgpt_smpc),
                       "scGPT": get_cent_performance(ds, os.path.join(root_dir, ds, 'centralized', 'results.pkl'))}

    rows = []

    for dataset, modes in results.items():
        for mode, metrics in modes.items():
            for metric, value in metrics.items():
                rows.append({
                    'Dataset': dataset,
                    'Mode': mode,
                    'Metric': metric,
                    'Value': value
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
        results[metric] = df.loc[idmax, "Value"]
    return results


def handle_metrics(metric):
    metric = metric[5:]
    if metric == "macro_f1":
        return 'Macro F1'
    return metric.title()

def plot_best_metrics(root_dir, param_tuning_df, img_format='svg'):
    handle_image_format(img_format)
    df = create_metrics_dataframe(root_dir, param_tuning_df)
    # TODO: check it works for new approach titles
    best_metrics_report(df)

    # Plot metrics with subplots for each metric and different modes as curves
    metrics = df['Metric'].unique()
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 4))


    for i, metric in enumerate(metrics):
        ax = axes[i] if num_metrics > 1 else axes
        sns.barplot(data=df[df['Metric'] == metric], x='Dataset', y='Value', hue='Mode', ax=ax, width=0.4)
        ax.set_ylabel(metric, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        dataset_names = df['Dataset'].unique()
        ax.set_xticklabels([handle_ds_name(ds) for ds in dataset_names], fontsize=16)

    # Get the handles and labels from the last axis
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, [l.title() for l in labels], loc='upper left', bbox_to_anchor=(0.05, 0.98), fontsize=16, ncol=2)

    # Remove legends from all axes
    for ax in axes:
        ax.get_legend().remove()
    legend_offset = 0.12
    plt.tight_layout(rect=[0, 0, 1 - legend_offset, 1 -legend_offset])  # Adjust rect to make space for the legend
    plt.savefig(f"{ANNOTATION_PLOTS_DIR}/best_metrics.{img_format}", dpi=300, format=img_format)
    plt.close()
    handle_image_format(start=False)


def best_metrics_report(df):
    # Calculate the differences between federated and centralized
    df_pivot = df.pivot_table(index=['Dataset', 'Metric'], columns='Mode', values='Value').reset_index()
    df_pivot['Difference'] = df_pivot['federated'] - df_pivot['centralized']
    # Calculate the percentage of centralized performance achieved by federated learning
    df_pivot['Percentage Achieved'] = (df_pivot['federated'] / df_pivot['centralized']) * 100
    # Print the difference and percentage for each metric
    print("Differences and Percentage Achieved between Federated and Centralized for each metric:")
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
    # plot_embedding_boxplot(df, img_format)
    display_federated_performance_report(df)
    # per_metric_annotated_scatterplot(df, "./plots/embedding", img_format)

def find_federated_performance_comparison(df, federated_types=None):
    """
    Identify worst federated vs. centralized performance for each federated type in federated_types,
    categorize equal/higher/lower, and compute minimum percentage relative to centralized scGPT.

    Returns a dict mapping each federated type to its report tuple:
    (worst_report, higher_list, equal_list, lower_list, N)
    """
    if federated_types is None:
        federated_types = ['FedscGPT', 'FedscGPT-SMPC']
    results = {}
    for fed in federated_types:
        # Pivot to compare centralized (scGPT) vs this federated type
        performance_diff = (
            df[df['Type'].isin(['scGPT', fed])]
            .pivot_table(index=['Dataset','Metric'], columns='Type', values='Value')
            .reset_index()
        )
        performance_diff['scGPT'] = performance_diff['scGPT'].round(3)
        performance_diff[fed] = performance_diff[fed].round(3)
        performance_diff['Difference'] = (performance_diff[fed] - performance_diff['scGPT']).round(3)
        performance_diff['Percentage'] = (performance_diff[fed] / performance_diff['scGPT'] * 100).round(2)

        N = performance_diff['Percentage'].min()
        worst = performance_diff.loc[performance_diff['Difference'].idxmin()]
        worst_report = (
            f"Worst performance of {fed} vs scGPT is on '{worst['Dataset']}' for metric '{worst['Metric']}', "
            f"Difference={worst['Difference']:.3f} ({fed}={worst[fed]:.3f}, scGPT={worst['scGPT']:.3f})."
        )

        higher = performance_diff[performance_diff['Difference'] > 0][['Dataset','Metric',fed,'scGPT']].values.tolist()
        equal = performance_diff[performance_diff['Difference'] == 0][['Dataset','Metric',fed,'scGPT']].values.tolist()
        lower = performance_diff[performance_diff['Difference'] < 0][['Dataset','Metric',fed,'scGPT']].values.tolist()

        results[fed] = (worst_report, higher, equal, lower, N)
    return results


def display_federated_performance_report(df):
    """
    Print performance comparisons for both FedscGPT and FedscGPT-SMPC against scGPT.
    """
    reports = find_federated_performance_comparison(df)
    for fed, (worst_report, higher_list, equal_list, lower_list, N) in reports.items():
        print(f"=== Performance Comparison: {fed} vs scGPT ===")
        print(worst_report)
        print(f"{fed}, in a privacy-aware federated manner, consistently reached at least {N}% of scGPT's performance across all metrics and datasets.")

        if higher_list:
            print(f"{fed} performed better than scGPT in:")
            for ds, metric, p_fed, p_cen in higher_list:
                print(f" - {ds} | {metric}: {fed}={p_fed:.3f} > scGPT={p_cen:.3f}")
        if equal_list:
            print(f"{fed} performed equally to scGPT in:")
            for ds, metric, p_fed, p_cen in equal_list:
                print(f" - {ds} | {metric}: {fed}={p_fed:.3f} = scGPT={p_cen:.3f}")
        if lower_list:
            print(f"{fed} performed worse than scGPT in:")
            for ds, metric, p_fed, p_cen in lower_list:
                print(f" - {ds} | {metric}: {fed}={p_fed:.3f} < scGPT={p_cen:.3f}")
    worst_report, higher_list, equal_list, lower_list, N = find_federated_performance_comparison(df)

    print(worst_report)
    print(f"\nFedscGPT, in a privacy-aware federated manner, consistently reached at least {N}% of scGPT's performance across all metrics and datasets.")
    if higher_list:
        print("\nFederated model performed better in:")
        for ds, metric, fed, cen in higher_list:
            print(f" - {ds} | {metric}: FedscGPT={fed:.3f} > scGPT={cen:.3f}")
    if equal_list:
        print("\nFederated model performed equally in:")
        for ds, metric, fed, cen in equal_list:
            print(f" - {ds} | {metric}: FedscGPT={fed:.3f} = scGPT={cen:.3f}")
    if lower_list:
        print("\nFederated model performed worse in:")
        for ds, metric, fed, cen in lower_list:
            print(f" - {ds} | {metric}: FedscGPT={fed:.3f} < scGPT={cen:.3f}")


def per_metric_annotated_scatterplot(df, plots_dir, img_format='svg', proximity_threshold=0.1):
    """
    Plot data using Matplotlib from a pandas DataFrame, with each scatter point annotated by its corresponding 'Batch' value.
    Adjusts the text to the left or right dynamically to avoid overlap for points with close y-values.

    Parameters:
    - df: DataFrame containing the data to plot.
    - plots_dir: Directory to save the plots.
    - img_format: Format to save the images (e.g., 'svg').
    - proximity_threshold: Defines the closeness of y-values to consider them overlapping (default = 0.1).
    """
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    metrics = df['Metric'].unique()
    datasets = df['Dataset'].unique()

    for metric in metrics:
        plt.figure(figsize=(5, 5))

        # Separate client data and centralized/federated data
        client_data = df[(df['Metric'] == metric) & (df['Type'].str.contains('client'))]
        centralized_data = df[(df['Metric'] == metric) & (df['Type'] == 'Centralized')]
        federated_data = df[(df['Metric'] == metric) & (df['Type'] == 'Federated')]

        # Scatter plot for client data
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgrey', 'lightyellow']  # Extend as needed
        for i, dataset in enumerate(datasets):
            dataset_clients = client_data[client_data['Dataset'] == dataset]
            client_values = dataset_clients['Value'].values
            client_batches = dataset_clients['Batch'].values
            # Scatter each client point with a slight horizontal offset to avoid overlap
            jitter = 0.05  # Add some horizontal jitter to avoid overlap
            x_jitter = np.random.uniform(-jitter, jitter, size=client_values.shape)
            scatter = plt.scatter([i + 1 + x for x in x_jitter], client_values,
                                  color=colors[i % len(colors)], edgecolor='black', s=50, alpha=0.7)

            # Determine proximity of y-values to decide label positions
            for j, (x, y, batch) in enumerate(zip([i + 1 + x for x in x_jitter], client_values, client_batches)):
                batch_label = shorten_batch_value(batch)

                # Check if other points are "close enough" in y-value using the proximity threshold
                annot_fontsize = 12
                close_points = np.sum(np.abs(client_values - y) < proximity_threshold)
                if close_points > 1:  # If there are other points within the threshold range
                    # Alternate placement of labels for overlapping points
                    if j % 2 == 0:
                        plt.text(x + 0.05, y, batch_label, fontsize=annot_fontsize, ha='left',
                                 va='center')  # Place to the right
                    else:
                        plt.text(x - 0.05, y, batch_label, fontsize=annot_fontsize, ha='right',
                                 va='center')  # Place to the left
                else:
                    plt.text(x + 0.05, y, batch_label, fontsize=annot_fontsize, ha='left',
                             va='center')  # Place to the right by default

        # Overlay centralized and federated data points
        for i, dataset in enumerate(datasets):
            # Centralized as horizontal lines only within the dataset range
            if not centralized_data[centralized_data['Dataset'] == dataset].empty:
                centralized_value = centralized_data[centralized_data['Dataset'] == dataset]['Value'].values[0]
                plt.hlines(y=centralized_value, xmin=i + 0.7, xmax=i + 1.3,
                           color=colors[i % len(colors)], linestyle='--', linewidth=2, zorder=3)

            # Federated as scatter points
            if not federated_data[federated_data['Dataset'] == dataset].empty:
                federated_value = federated_data[federated_data['Dataset'] == dataset]['Value'].values[0]
                plt.scatter(i + 1, federated_value, color=colors[i % len(colors)], edgecolor='black',
                            zorder=5, marker='D', s=100)

        # Customize the plot
        plt.xlabel('', fontsize=1)
        plt.ylabel(metric.capitalize(), fontsize=20)
        plt.xticks(range(1, len(datasets) + 1), [handle_ds_name(d) for d in datasets], fontsize=20)
        plt.yticks(fontsize=18)

        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{metric}_scatterplot_annotated.{img_format}", format=img_format, dpi=300,
                    bbox_inches='tight')
        plt.close()
        plot_legend(plots_dir, img_format)

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
                        color=box['boxes'][i].get_facecolor(), edgecolor='black', zorder=5, marker='D', s=100)
            # FedscGPT-SMPC as scatter points
            plt.scatter(i + 1, fedscgpt_smpc[fedscgpt_smpc['Dataset'] == dataset]['Value'].values[0],
                        color=box['boxes'][i].get_facecolor(), edgecolor='black', zorder=5, marker='*', s=100)

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
            Line2D([0], [0], marker='*', color='w', markersize=10, label='FedscGPT',
                   markeredgecolor='black'),
            Line2D([0], [0], marker='D', color='w', markersize=10, label='FedscGPT-SMPC',
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