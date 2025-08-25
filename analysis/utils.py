from sympy.geometry.entity import rotate

import __init__
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
from pathlib import Path
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from data.stats import datasets as datasets_details
from data.stats import batch_map, celltype_mapping
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

arial_fp = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
arial_prop = fm.FontProperties(fname=arial_fp)
import matplotlib.font_manager as fm
import matplotlib.font_manager as fm

arial_fp = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
fm.fontManager.addfont(arial_fp)
plt.rcParams['font.family'] = fm.FontProperties(fname=arial_fp).get_name()

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

set_seed(SEED)
image_format = 'svg'
ANNOTATION_PLOTS_DIR = 'plots/annotation'
EMBEDDING_PLOTS_DIR = 'plots/embedding'
FEDSCGPT_MARKER = 'D'
FEDSCGPT_SMPC_MARKER = '*'

FEDAVG_MARKER = "D"
FEDAVG_SMPC_MARKER = "s"
FEDPROX_MARKER = "P"
FEDPROX_SMPC_MARKER = "X"

def print_config(config: dict or tuple, level=0):
    for k, v in config.items():
        if isinstance(v, dict):
            print("  " * level + str(k) + ":")
            print_config(v, level + 1)
        else:
            print("  " * level + str(k) + ":", v)

safe_extended_palette = [
    '#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc',
    '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9',
    '#0c69ff', '#49ff90', '#4b00bd', '#00cae4', '#bcdbff',
    '#856ef9', '#00441f', '#91a5ba', '#00c87e', '#0179aa'
]

def generate_palette(unique_celltypes):
    """
    Generate a red-green colorblind–friendly palette for cell type labels.

    Args:
        unique_celltypes: list of category names

    Returns:
        dict mapping each category to a color hex code
    """
    n_cats = len(unique_celltypes)
    palette_ = {}

    if n_cats <= len(safe_extended_palette):
        # Direct mapping from safe palette
        for i, cat in enumerate(unique_celltypes):
            palette_[cat] = safe_extended_palette[i]
    else:
        # Use all predefined safe colors
        for i, cat in enumerate(unique_celltypes[:len(safe_extended_palette)]):
            palette_[cat] = safe_extended_palette[i]

        # Generate additional colors from a perceptually uniform color map
        extra_colors = plt.get_cmap("cividis", n_cats - len(safe_extended_palette))
        for i, cat in enumerate(unique_celltypes[len(safe_extended_palette):]):
            rgba = extra_colors(i)
            # Convert to hex
            palette_[cat] = '#%02x%02x%02x' % tuple(int(255 * c) for c in rgba[:3])

    return palette_


DATASETS_DETAILS = {'ms': {'reference': 'reference_annot.h5ad',
                   'query': 'query_annot.h5ad',
                   'cell_type_key': 'Factor Value[inferred cell type - authors labels]',
                   'batch_key': 'split_label',},
    'hp5': {'reference': 'reference.h5ad',
           'query': 'query.h5ad',
           'cell_type_key': 'Celltype',
           'batch_key': 'batch_name'},
    'myeloid': {'reference': 'reference_adata.h5ad',
                'query': 'query_adata.h5ad',
                'cell_type_key': 'combined_celltypes',
                'batch_key': 'top4+rest'},
    'lung': {'reference': 'reference_annot.h5ad',
                'query': 'query_annot.h5ad',
                'cell_type_key': 'cell_type',
                'batch_key': 'sample'},
    'cl': {'reference': 'reference.h5ad',
                'query': 'query.h5ad',
                'cell_type_key': 'cell_type',
                'batch_key': 'batch'},
    'covid': {'reference': 'reference-raw.h5ad',
                'query': 'query-raw.h5ad',
                'cell_type_key': 'celltype',
                'batch_key': 'batch_group'},
    'covid-corrected': {'reference': 'reference_corrected.h5ad',
                        'query': 'query_corrected.h5ad',
                        'cell_type_key': 'celltype',
                        'batch_key': 'batch_group'},
    'covid-fed-corrected': {'reference': 'reference_fed_corrected.h5ad',
                            'query': 'query_fed_corrected.h5ad',
                            'cell_type_key': 'celltype',
                            'batch_key': 'batch_group'},
    'myeloid-top5': {'reference': 'reference.h5ad',
                        'query': 'query.h5ad',
                        'cell_type_key': 'cell_type',
                        'batch_key': 'combined_batch'},
    'myeloid-top10': {'reference': 'reference.h5ad',
                        'query': 'query.h5ad',
                        'cell_type_key': 'cell_type',
                        'batch_key': 'combined_batch'},
    'myeloid-top20': {'reference': 'reference.h5ad',
                        'query': 'query.h5ad',
                        'cell_type_key': 'cell_type',
                        'batch_key': 'combined_batch'},
    'myeloid-top30': {'reference': 'reference.h5ad',
                        'query': 'query.h5ad',
                        'cell_type_key': 'cell_type',
                        'batch_key': 'combined_batch'},
    }



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

def plot_stacked_fedavg_heatmaps(df, save_path="fedavg_stacked_heatmap", file_format='png'):
    """
    Plot heatmaps where each row is a metric (e.g., Accuracy, F1)
    and each column is a federated scenario (e.g., FedAvg on HP, etc.).
    """
    configs = [
        ("hp5", "weighted-FedAvg", "HP (FedAvg)"),
        ("hp5", "SMPC-weighted-FedAvg", "HP (SMPC-FedAvg)"),
        ("ms", "SMPC-weighted-FedAvg", "MS (SMPC-FedAvg)"),
        ("myeloid-top5", "SMPC-weighted-FedAvg", "Myeloid (SMPC-FedAvg)"),
    ]

    # Clean and restrict data
    df = df.dropna(subset=['Round', 'n_epochs'])
    df.Round = df.Round.astype(int)
    df.n_epochs = df.n_epochs.astype(int)
    df = df[df['Round'].between(1, 15) & df['n_epochs'].between(1, 5)]

    metrics = df['Metric'].unique()
    num_rows = len(metrics)
    num_cols = len(configs)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 1.), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.92, bottom=0.1, wspace=0.02, hspace=0.05)

    # Blue-white color map
    norm = colors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('Blues')
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    for row_idx, metric in enumerate(metrics):
        for col_idx, (dataset, agg, label) in enumerate(configs):
            ax = axs[row_idx][col_idx]
            subset = df[(df['Dataset'] == dataset) &
                        (df['Aggregation'] == agg) &
                        (df['Metric'] == metric)]
            if subset.empty:
                ax.set_title("No data", fontsize=10)
                ax.axis('off')
                continue
            subset = subset.drop_duplicates(subset=['Round', 'n_epochs', 'Metric', 'Aggregation', 'Dataset'], keep='last')
            pivot = subset.pivot(index='n_epochs', columns='Round', values='Value')
            sns.heatmap(
                pivot, ax=ax, cmap=cmap, cbar=False, center=0.5,
                vmin=0, vmax=1, square=True, linewidths=0.1, linecolor='gray', annot=False
            )

            # Annotate best
            max_row = subset.loc[subset['Value'].idxmax()]
            print(metric, dataset, max_row['Round'], max_row['n_epochs'])
            ax.text(
                max_row['Round'] - 1, max_row['n_epochs'] - 0.5,
                f"{max_row['Value']:.2f}".lstrip("0"),
                color='black', ha='left', va='center', fontsize=8
            )

            # Titles and axis labels
            if row_idx == 0:
                ax.set_title(label, fontsize=12)
            if col_idx == 0:
                # ax.set_ylabel(metric, fontsize=12)
                ax.text(-0.22, 0.15, metric, transform=ax.transAxes, fontsize=10, va='bottom', ha='left',
                        rotation=90)
                ax.set_ylabel("Epochs", fontsize=8)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([], visible=False)

            if row_idx < num_rows - 1:
                ax.set_xticklabels([], visible=False)
            else:
                ax.set_xlabel("Round", fontsize=10)

            # Custom ticks
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns.tolist(), fontsize=8, rotation=0)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index.tolist(), fontsize=8, rotation=0)

    # Add common colorbar (small & thin)
    cbar_ax = fig.add_axes([0.87, 0.3, 0.01, 0.4])
    cbar = plt.colorbar(mappable, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{save_path}.{file_format}", dpi=300, bbox_inches='tight')
    plt.close()



def plot_communication_efficiency(df):

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

def filter_and_export_metrics(
        xls,
        output_excel_path: str,
        dataset_name_map: dict
):
    """
    Combines all metric sheets into one table with rows as a combined (Approach, Dataset)
    column and columns as threshold levels (70%, 80%, 90%, 95%, 99%).

    Parameters:
    - xls: pd.ExcelFile
    - output_excel_path: str
    - dataset_name_map: dict mapping original names to display names
    """
    combined = None

    # Define threshold levels
    threshold_levels = ['70%', '80%', '90%', '95%', '99%']
    writer = pd.ExcelWriter(f"{ANNOTATION_PLOTS_DIR}/efficiency/{output_excel_path}", engine='xlsxwriter')
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        metric = df.Metric.unique()
        assert len(metric) == 1, f"Expected 1 metric, got {len(metric)}"
        metric = metric[0]
        if not {'Aggregation', 'Dataset', 'Min Round', 'Epoch', 'mu'}.issubset(df.columns):
            print(f"Skipping sheet {sheet_name} (missing columns)")
            continue

        # Filter & rename datasets
        df = df[df['Dataset'].isin(dataset_name_map.keys())].copy()
        df['Dataset'] = df['Dataset'].map(dataset_name_map)

        # Filter for SMPC entries and rename aggregations
        df = df[df['Aggregation'].str.contains('SMPC')]
        agg_map = {"SMPC-weighted-FedAvg": "FedAvg-SMPC",
                   "SMPC-weighted-FedProx": "FedProx-SMPC"}
        df['Aggregation'] = df['Aggregation'].replace(agg_map)

        # Combine Aggregation and Dataset into a single column
        df['Combined'] = df['Dataset'] + '(' + df['Aggregation'] + ')'
        df = df.drop(columns=['Aggregation', 'Dataset'])

        # Pivot data to get threshold levels with rounds, epochs, and mu
        pivot_df = df.pivot_table(index='Combined',
                                columns='Threshold',
                                values=['Min Round', 'Epoch', 'mu'],
                                aggfunc='first',
                                fill_value='NR').reset_index()

        # Restore NaN for 'mu' where it was filled with 'NR'
        for level in threshold_levels:
            if ('mu', level) in pivot_df.columns:
                pivot_df[('mu', level)] = pivot_df[('mu', level)].replace('NR', np.nan)

        # Combine Min Round, Epoch, and mu into a single string with pipe separator
        for level in threshold_levels:
            if (('Min Round', level) in pivot_df.columns and ('Epoch', level) in pivot_df.columns and ('mu', level) in pivot_df.columns):
                pivot_df[level] = pivot_df.apply(
                    lambda row: f"{row[('Min Round', level)]}|{row[('Epoch', level)]}|{row[('mu', level)]}" if pd.notna(row[('Min Round', level)]) and pd.notna(row[('Epoch', level)]) and pd.notna(row[('mu', level)]) else
                               f"{row[('Min Round', level)]}|{row[('Epoch', level)]}" if pd.notna(row[('Min Round', level)]) and pd.notna(row[('Epoch', level)]) else 'NR',
                    axis=1
                )
            else:
                pivot_df[level] = 'NR'

        # Replace 'NR|NR' and 'NR|NR|NR' with 'NR'
        for level in threshold_levels:
            pivot_df[level] = pivot_df[level].replace('NR|NR', 'NR').replace('NR|NR|NR', 'NR')

        # Drop the multi-level columns for Min Round, Epoch, and mu
        pivot_df = pivot_df.drop(columns=[('Min Round', lvl) for lvl in threshold_levels] + [('Epoch', lvl) for lvl in threshold_levels] + [('mu', lvl) for lvl in threshold_levels])

        # Reorder columns to match figure format
        cols = ['Combined'] + threshold_levels
        pivot_df = pivot_df[cols]
        pivot_df = pivot_df.sort_values(by='Combined')
        pivot_df.columns = [
            '_'.join(str(c) for c in col).strip('_') if isinstance(col, tuple) else col
            for col in pivot_df.columns
        ]
        pivot_df.rename(columns={'Combined': 'Dataset (Aggregation)'}, inplace=True)
        pivot_df.to_excel(writer, sheet_name=metric, index=False)
        worksheet = writer.sheets[metric]  # same sheet name you wrote above

        # Step 2: Define formats
        red_fmt = writer.book.add_format({'font_color': 'red'})
        black_fmt = writer.book.add_format({'font_color': 'black'})

        # Step 3: Loop over the written DataFrame and rewrite rich strings
        for row_idx in range(1, len(pivot_df) + 1):  # skip header row
            for col_idx in range(1, len(threshold_levels) + 1):  # skip "Dataset (Aggregation)"
                cell_text = str(pivot_df.iloc[row_idx - 1, col_idx])

                if '|' in cell_text and cell_text != 'NR':
                    parts = []
                    for i, chunk in enumerate(cell_text.split('|')):
                        if i > 0:
                            parts.append(red_fmt)
                            parts.append('|')
                        parts.append(black_fmt)
                        parts.append(chunk)
                    worksheet.write_rich_string(row_idx, col_idx, *parts)

        print(pivot_df)
        fig, ax = plt.subplots(figsize=(10, len(pivot_df) * 0.3))  # Adjust figsize based on data
        ax.axis('off')  # Hide axes
        table = ax.table(cellText=pivot_df.values, colLabels=pivot_df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)  # Adjust scale for readability
        plt.savefig(f'{ANNOTATION_PLOTS_DIR}/efficiency/{sheet_name}_table.png', dpi=300, bbox_inches='tight')
        plt.close()
    writer.close()


import pandas as pd
import os

def export_optimal_params_by_metric(xls, dataset_name_map):
    """
    Exports the best parameters for each federated scenario, organized by metric into separate sheets
    in an XLSX file. Combines Dataset and Aggregation, removes 'weighted-' from Aggregation,
    and incorporates mu into Aggregation where applicable.

    Parameters:
    - xls: pd.ExcelFile
    - dataset_name_map: dict mapping original names to display names
    """
    output_file = os.path.join(ANNOTATION_PLOTS_DIR, "efficiency", "best_params.xlsx")
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # Extract metric from sheet name (e.g., "Accuracy" from "sheet_Accuracy")
        metric = "-".join(sheet_name.split('_')[1:])
        print(f"Processing metric: {metric}")

        # Filter and rename datasets
        df = df[df['Dataset'].isin(dataset_name_map.keys())].copy()
        df['Dataset'] = df['Dataset'].map(dataset_name_map)

        # Ensure all required columns are present
        required_columns = {'Dataset', 'Aggregation', 'Best Value', 'n_epochs', 'mu', 'Round'}
        if not required_columns.issubset(df.columns):
            print(f"Skipping sheet {sheet_name} (missing required columns)")
            continue

        # Combine Dataset and Aggregation, remove 'weighted-', and incorporate mu
        df['Aggregation'] = df['Aggregation'].str.replace('weighted-', '')
        df['Aggregation'] = df['Aggregation'].apply(lambda x: f"{x[5:]}-SMPC" if x.startswith('SMPC-') else x)
        df['Combined'] = df['Dataset'] + ' (' + df['Aggregation'] + df['mu'].apply(lambda x: f", μ={x}" if pd.notna(x) else '') + ')'
        df = df.drop(columns=['Dataset', 'Aggregation', 'mu'])

        # Reorder and select relevant columns
        df = df[['Combined', 'Best Value', 'n_epochs', 'Round']]
        df['Best Value'] = df['Best Value'].astype(float).round(3)  # Round to 3 decimal places
        df.rename(columns={'Combined': 'Dataset (Aggregation)'}, inplace=True)
        # Create a sheet for this metric
        df.to_excel(writer, sheet_name=metric, index=False)
        print(f"Exported data for metric {metric} to sheet: {metric}")
        print(df)

    # Save the Excel file
    writer.close()
    print(f"Best parameters exported to: {output_file}")





def plot_metric_changes_over_ER(df, dataset_name_map, epochs_list, target_metric='Accuracy', img_format='svg'):
    """
    Plots how a target metric changes over communication rounds for different epoch settings.

    Parameters:
    - df: pandas DataFrame with ['Round', 'n_epochs', 'Dataset', 'Metric', 'Value']
    - dataset_name_map: dict, mapping raw dataset names to pretty labels
    - epochs_list: list of ints, which epochs to plot (e.g., [1, 5, 10])
    - target_metric: str, metric to plot (e.g., 'Accuracy')
    - img_format: str, file format to save (e.g., 'svg', 'png')
    """
    df = df[df.Dataset.isin(dataset_name_map.keys())]
    df = df[df.Aggregation == "SMPC-weighted-FedAvg"]
    # Filter and cast
    df = df[df['Round'] < 11].copy()
    df['Round'] = df['Round'].astype(int)
    df['n_epochs'] = df['n_epochs'].astype(int)

    # Map dataset names
    df['Dataset'] = df['Dataset'].map(dataset_name_map)

    datasets = df['Dataset'].unique()
    num_datasets = len(datasets)

    fig, axs = plt.subplots(1, num_datasets, figsize=(3 * num_datasets, 3), sharey=True)
    if num_datasets == 1:
        axs = [axs]

    font_properties = {'fontsize': 12}
    xlim = (df['Round'].min(), df['Round'].max())

    for i, dataset in enumerate(datasets):
        ax = axs[i]
        r_zero = df[(df['Dataset'] == dataset) & (df['Round'] == 0) & (df['Metric'] == target_metric)].Value.values[0]
        for n_epochs in epochs_list:
            data = df[
                (df['Dataset'] == dataset) &
                (df['n_epochs'] == n_epochs) &
                (df['Metric'] == target_metric)
            ].sort_values(by='Round')
            if 0 not in data.Round.unique():
                # Add a row for Round 0 with the initial value
                data = pd.concat([pd.DataFrame({'Round': [0], 'Value': [r_zero], 'n_epochs': [n_epochs], 'Dataset': [dataset], 'Metric': [target_metric]}), data])


            sns.lineplot(data=data, x='Round', y='Value', label=f'{n_epochs} Epochs', marker='o', ax=ax)

        ax.set_title(dataset, **font_properties)
        ax.set_xlim(xlim)
        ax.set_ylim((0, 1))
        ax.set_xlabel('Rounds', **font_properties)
        ax.set_xticks(list(range(1, int(xlim[1]) + 1)))

        if i == 0:
            ax.set_ylabel(target_metric, **font_properties)
        else:
            ax.set_ylabel('')

        ax.grid(True)
        ax.legend().set_visible(False)

    # Add shared legend
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.94), ncol=len(epochs_list), fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.8, wspace=0.051)

    output_path = f"{ANNOTATION_PLOTS_DIR}/efficiency/{target_metric}_changes_over_ER.{img_format}"
    plt.savefig(output_path, format=img_format, dpi=300)
    print(f"Saved plot to {output_path}")



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
    if name.lower() == "hp5":
        return "HP"
    if name.lower() == "ms":
        return "MS"
    if name.lower() == "ms-corrected":
        return "MS-Corrected"
    if name.lower() == "ms-fed-corrected":
        return "MS-Fed-Corrected"
    if name.lower() == "myeloid":
        return "Myeloid"
    if name.lower() == "myeloid-top5":
        return "Myeloid"
    if name.lower() == "covid":
        return "Covid-19"
    if name.lower() == "lung":
        return "Lung-Kim"
    if name.lower() == "cl":
        return "CL"
    return name.split("-")[-1]

DS_NAME_MAP = {
    "hp": "HP",
    "ms": "MS",
    "myeloid": "Myeloid",
    "covid": "Covid-19",
    "lung": "Lung-Kim",
    "cl": "CL",
}


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


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import anndata
from sklearn.metrics import confusion_matrix
from matplotlib.colors import Normalize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_and_print_metrics(labels, cent_preds, fed_preds, unique_ct, average='weighted'):
    # print(f"Evaluating {average} average with {'Other in' if 'Other' in unique_ct else 'all'} unique cell types...")
    accuracy_cent = round(accuracy_score(labels, cent_preds), 2)
    precision_cent = round(precision_score(labels, cent_preds, labels=unique_ct, average=average, zero_division=0), 2)
    recall_cent = round(recall_score(labels, cent_preds, labels=unique_ct, average=average, zero_division=0), 2)
    f1_cent = round(f1_score(labels, cent_preds, labels=unique_ct, average=average, zero_division=0), 2)

    # Federated metrics
    accuracy_fed = round(accuracy_score(labels, fed_preds), 2)
    precision_fed = round(precision_score(labels, fed_preds, labels=unique_ct, average=average, zero_division=0), 2)
    recall_fed = round(recall_score(labels, fed_preds, labels=unique_ct, average=average, zero_division=0), 2)
    f1_fed = round(f1_score(labels, fed_preds, labels=unique_ct, average=average, zero_division=0), 2)


    print(f"[Centralized] Accuracy: {accuracy_cent}, Precision: {precision_cent}, Recall: {recall_cent}, F1: {f1_cent}")
    print(f"[Federated]   Accuracy: {accuracy_fed}, Precision: {precision_fed}, Recall: {recall_fed}, F1: {f1_fed}")



def plot_confusion_with_deltas(dataset, cent_preds, fed_preds, labels, id_maps, color_mapping, save_path):
    labels = [id_maps[c] for c in labels]
    cent_preds = [id_maps[c] for c in cent_preds]
    fed_preds = [id_maps[c] for c in fed_preds]
    unique_ct = list(color_mapping.keys())
    evaluate_and_print_metrics(labels, cent_preds, fed_preds, unique_ct=None, average='macro')
    # assert labels are in color_mapping keys
    assert set(cent_preds) - set(unique_ct) == set(), \
        f" Centralized predictions {set(cent_preds) - set(unique_ct)} are not in color mapping keys"
    assert set(fed_preds) - set(unique_ct) == set(), \
        f" Federated predictions {set(fed_preds) - set(unique_ct)} are not in color mapping keys"

    cm_centralized = confusion_matrix(labels, cent_preds, labels=unique_ct)
    cm_federated = confusion_matrix(labels, fed_preds, labels=unique_ct)

    # Normalize by row (i.e., per ground truth cell type), handling division by zero
    with np.errstate(invalid='ignore', divide='ignore'):
        cm_centralized = cm_centralized.astype("float") / cm_centralized.sum(axis=1, keepdims=True)
        cm_federated = cm_federated.astype("float") / cm_federated.sum(axis=1, keepdims=True)

    # Replace NaNs (from division by zero) with 0
    cm_centralized = np.nan_to_num(cm_centralized)
    cm_federated = np.nan_to_num(cm_federated)

    # Compute difference matrix (delta between federated and centralized)
    diff = cm_federated - cm_centralized

    # Prepare empty annotation matrix for superscript deltas
    annot = np.full(diff.shape, "", dtype=object)



    # Annotate only when difference is non-zero
    # diff = cm_federated - cm_centralized
    # annot = np.full(diff.shape, "", dtype=object)
    for i in range(cm_federated.shape[0]):
        for j in range(cm_federated.shape[1]):
            base = cm_federated[i, j]
            delta = diff[i, j]

            # Format base value
            if base < 0.005:
                base_str = ""
            elif base > 0.995:
                base_str = "1"
            else:
                base_str = f"{base:.2f}".lstrip("0")
            annot[i, j] = base_str
            # Format diff as superscript only if non-zero
            sign = "+" if delta > 0 else "-"
            delta_str = f"{abs(delta):.2f}".lstrip("0")
            delta_str = f"{sign}{delta_str}"
            annot[i, j] = ""
            if base_str == "":
                if delta_str != "-.00" and delta_str != "+.00":
                    annot[i, j] = f"$0^{{{delta_str}}}$"
            else:
                if delta_str == "-.00" or delta_str == "+.00":
                    annot[i, j] = base_str
                else:
                    annot[i, j] = f"${base_str}^{{{delta_str}}}$"

    df_cm = pd.DataFrame(cm_federated, index=unique_ct, columns=unique_ct)
    print("Diagonal entries with Δ differences (Federated vs Centralized):")
    for i in range(len(unique_ct)):
        print(f"{unique_ct[i]:<25}: {annot[i, i]}")

    # fig_width = len(unique_ct)
    # fig_height = len(unique_ct)
    # fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # sns_heatmap = sns.heatmap(df_cm, annot=annot, fmt='', cmap='Blues', cbar=False, square=True,
    #                           annot_kws={"fontsize": 18})
    #
    # ax = plt.gca()
    # ax.set_xticks(np.arange(len(df_cm.columns)) + 0.5)
    # ax.set_yticks(np.arange(len(df_cm.index)) + 0.5)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.tick_params(axis='both', which='both', length=0)
    #
    # # Color tick boxes
    # for ytick, label in zip(ax.get_yticks(), df_cm.index):
    #     color = color_mapping.get(label, 'black')
    #     rect = plt.Rectangle((-0.02, ytick - 0.5), 0.02, 1, color=color,
    #                          transform=ax.get_yaxis_transform(), clip_on=False)
    #     ax.add_patch(rect)
    #
    # for xtick, label in zip(ax.get_xticks(), df_cm.columns):
    #     color = color_mapping.get(label, 'black')
    #     rect = plt.Rectangle((xtick - 0.5, 1.0), 1, 0.02, color=color,
    #                          transform=ax.get_xaxis_transform(), clip_on=False)
    #     ax.add_patch(rect)
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=300)
    # plt.close()


def find_best_macro_f1(dataset, df, fed_aggregations, summary):
    folder = 'myeloid-top30' if 'myeloid' in dataset else dataset
    cent = summary[folder]['centralized'][20][None][None]['predictions']
    labels = summary[folder]['centralized'][20][None][None]['labels']
    cent_f1 = df[(df.Aggregation == 'centralized') & (df.Dataset == dataset) & (df.Metric == 'Macro_F1') & (
            df.n_epochs == 20)]
    assert len(cent_f1) == 1, f"There should be only one row for centralized macro F1\n {cent_f1}"
    cent_f1 = cent_f1.Value.values[0]
    best_macro_f1_idx = df[
        (df.Aggregation.isin(fed_aggregations)) & (df.Dataset == dataset) & (
                df.Metric == 'Macro_F1')].Value.idxmax()
    best_f1 = df.iloc[best_macro_f1_idx]['Value']
    best_agg = df.iloc[best_macro_f1_idx]['Aggregation']
    best_epoch = df.iloc[best_macro_f1_idx]['n_epochs']
    best_round = df.iloc[best_macro_f1_idx]['Round']
    best_mu = df.iloc[best_macro_f1_idx]['mu']
    best_mu = None if np.isnan(best_mu) else best_mu
    fed = summary[dataset][best_agg][best_epoch][best_round][best_mu]['predictions']
    id_maps = summary[dataset]['id_maps']
    filename = '{}_{}_E{}_R{}_MU{}_F1{}_CentF1{}'.format(dataset, best_agg, best_epoch, best_round, best_mu,
                                                         best_f1, cent_f1)
    return cent, fed, filename, id_maps, labels


def find_best_by_metric(
    dataset: str,
    df: pd.DataFrame,
    fed_aggregations: list[str],
    summary: dict,
    metric: str = "Macro_F1",
    centralized_epochs: int = 20
):
    """
    Selects the best federated run by `metric` and returns (centralized_preds, best_fed_preds, filename, id_maps, labels).

    Assumptions:
      - df has columns: ['Aggregation','Dataset','Metric','Value','n_epochs','Round','mu']
      - summary structure: summary[dataset][agg][epochs][round][mu]['predictions'|'labels']
      - centralized stored under summary[folder]['centralized'][centralized_epochs][None][None]
      - 'myeloid-top30' folder special-casing preserved
    """
    # Resolve folder naming convention
    folder = 'myeloid-top30' if 'myeloid' in dataset.lower() else dataset

    # Pull centralized refs
    cent_block = summary[folder]['centralized'][centralized_epochs][None][None]
    cent_preds = cent_block['predictions']
    labels = cent_block['labels']

    # Centralized metric value (optional but handy in filename)
    cent_row = df[
        (df['Aggregation'] == 'centralized') &
        (df['Dataset'] == dataset) &
        (df['Metric'] == metric) &
        (df['n_epochs'] == centralized_epochs)
    ]
    if len(cent_row) != 1:
        raise ValueError(f"Expected exactly one centralized row for {metric} @ {centralized_epochs} epochs; got:\n{cent_row}")
    cent_val = float(cent_row['Value'].iloc[0])

    # Candidate federated rows for the chosen metric
    mask = (
        df['Aggregation'].isin(fed_aggregations) &
        (df['Dataset'] == dataset) &
        (df['Metric'] == metric)
    )
    cand = df.loc[mask].copy()

    if cand.empty:
        raise ValueError(f"No federated rows found for dataset='{dataset}', metric='{metric}'.")

    # Prefer higher-is-better (F1/Recall/Precision/Accuracy). If you ever pass a loss, invert first.
    # Drop NaNs in Value to avoid idxmax errors
    cand = cand.dropna(subset=['Value'])

    # Break ties deterministically: higher Value, then more rounds, then more epochs (or tweak as you like)
    cand = cand.sort_values(by=['Value', 'Round', 'n_epochs'], ascending=[False, False, False])
    best_row = cand.iloc[0]

    best_agg  = best_row['Aggregation']
    best_val  = float(best_row['Value'])
    best_ep   = int(best_row['n_epochs'])
    best_rd   = best_row['Round']
    best_mu   = best_row.get('mu', np.nan)
    best_mu   = None if pd.isna(best_mu) else best_mu

    # Fetch federated predictions
    fed_preds = summary[dataset][best_agg][best_ep][best_rd][best_mu]['predictions']

    id_maps = summary[dataset]['id_maps']

    # Build filename tag
    fname = f"{dataset}_{best_agg}_E{best_ep}_R{best_rd}_MU{best_mu}_{metric}{best_val:.3f}_Cent{metric}{cent_val:.3f}"
    print(f"Best {metric} for {dataset}: {best_val}: \n\t Aggregation: {best_agg}\n\tepochs: {best_ep}\n\tRounds{best_rd}\n\tMU: {best_mu}) ")

    return cent_preds, fed_preds, fname, id_maps, labels

def best_perf_table(df):
    dropp_ds = ['myeloid', 'hp', 'covid', 'covid-fed-corrected']
    df = df[~df['Dataset'].isin(dropp_ds)]

    fed_aggs = ['FedAvg', 'FedAvg-SMPC', 'FedProx', 'FedProx-SMPC']
    fed_agg_map = {
        'SMPC-weighted-FedAvg': 'FedAvg-SMPC',
        'SMPC-weighted-FedProx': 'FedProx-SMPC'
    }
    df['Aggregation'] = df['Aggregation'].apply(lambda x: fed_agg_map.get(x, x))
    ds_map = {'hp5': 'HP', 'ms': 'MS', 'myeloid-top5': 'Myeloid-Top5', 'myeloid-top10': 'Myeloid-Top10',
              'myeloid-top20': 'Myeloid-Top20', 'myeloid-top30': 'Myeloid-Top30', 'covid-corrected': 'Covid-Corrected',
              'cl': 'CL', 'lung': 'Lung-Kim', }
    df['Dataset'] = df['Dataset'].map(ds_map)


    def format_agg_with_params(row):
        params = []
        if not pd.isna(row.get("n_epochs")):
            params.append(f"E={int(row['n_epochs'])}")
        if not pd.isna(row.get("Round")):
            params.append(f"R={int(row['Round'])}")
        if not pd.isna(row.get("mu")):
            params.append(f"Mu={row['mu']:.2f}")
        return f"{row['Aggregation']} ({', '.join(params)})" if params else row['Aggregation']

    metrics = df['Metric'].unique()
    output_excel = 'best_federated_vs_centralized.xlsx'

    with pd.ExcelWriter(f"{ANNOTATION_PLOTS_DIR}/summary/{output_excel}", engine='xlsxwriter') as writer:
        for metric in metrics:
            rows = []
            df_metric = df[df['Metric'] == metric]

            for dataset in df_metric['Dataset'].unique():
                df_dataset = df_metric[df_metric['Dataset'] == dataset]

                # Centralized value
                df_central = df_dataset[df_dataset['Aggregation'] == 'centralized']
                if df_central.empty:
                    continue
                centralized_val = df_central['Value'].values[0]

                # Federated subset
                df_fed = df_dataset[df_dataset['Aggregation'].isin(fed_aggs)]
                if df_fed.empty:
                    continue

                # Best federated
                idx_best = df_fed['Value'].idxmax()
                best_row = df.loc[idx_best].copy()

                rows.append({
                    'Dataset': dataset,
                    'Aggregation': format_agg_with_params(best_row),
                    'Federated': best_row['Value'].round(2),
                    'Centralized': centralized_val.round(2),
                    'Difference': (best_row['Value'] - centralized_val).round(2),
                })

            # Save to Excel
            df_result = pd.DataFrame(rows)
            df_result['Dataset'] = pd.Categorical(df_result['Dataset'],
                                                  categories=['CL', 'Covid-Corrected', 'HP', 'Lung-Kim', 'MS',
                                                              'Myeloid-Top5', 'Myeloid-Top10', 'Myeloid-Top20',
                                                              'Myeloid-Top30'], ordered=True)
            df_result.sort_values('Dataset', ascending=True, inplace=True)
            print(df_result)
            df_result.to_excel(writer, sheet_name=metric[:31], index=False)


def best_perf_table_reference_mapping(df):
    df = df[df['Aggregation'].isin(['smpc', 'centralized'])]
    ds_map = {
        'hp5': 'HP', 'ms': 'MS', 'myeloid-top5': 'Myeloid-Top5', 'myeloid-top10': 'Myeloid-Top10',
        'myeloid-top20': 'Myeloid-Top20', 'myeloid-top30': 'Myeloid-Top30',
        'covid-corrected': 'Covid-Corrected', 'cl': 'CL', 'lung': 'Lung-Kim',
    }
    df['Dataset'] = df['Dataset'].map(ds_map)

    # Define dataset order
    dataset_order = ['CL', 'Covid-Corrected', 'HP', 'Lung-Kim', 'MS',
                     'Myeloid-Top5', 'Myeloid-Top10', 'Myeloid-Top20', 'Myeloid-Top30']

    # Output file
    output_excel = 'best_ref_mapping_vs_centralized.xlsx'
    metrics = [ 'Macro-F1', 'Accuracy', 'Precision', 'Recall']

    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        for metric in metrics:
            rows = []

            for dataset in df['Dataset'].unique():
                df_dataset = df[df['Dataset'] == dataset]

                df_central = df_dataset[df_dataset['Aggregation'] == 'centralized']
                df_smpc = df_dataset[df_dataset['Aggregation'] == 'smpc']

                if df_central.empty or df_smpc.empty:
                    continue

                centralized_val = df_central[metric].values[0]
                smpc_val = df_smpc[metric].max()

                rows.append({
                    'Dataset': dataset,
                    'Federated (SMPC)': round(smpc_val, 2),
                    'Centralized': round(centralized_val, 2),
                    'Difference': round(smpc_val - centralized_val, 2)
                })

            df_result = pd.DataFrame(rows)
            df_result['Dataset'] = pd.Categorical(df_result['Dataset'], categories=dataset_order, ordered=True)
            df_result.sort_values('Dataset', inplace=True)

            print(df_result)
            df_result.to_excel(writer, sheet_name=metric[:31], index=False)



def plot_umap_and_conf_matrix(df, summary, data_dir):
    target_datasets = ['lung', 'cl', 'hp5', 'ms', 'myeloid-top5', 'myeloid-top10', 'myeloid-top20', 'myeloid-top30', 'covid', 'covid-corrected']
    fed_aggregations = ['SMPC-weighted-FedAvg', 'SMPC-weighted-FedProx']
    for dataset in target_datasets:
        print(dataset)
        cent, fed, filename, id_maps, labels = find_best_by_metric(dataset, df, fed_aggregations, summary, metric='Recall')

        query = anndata.read_h5ad(os.path.join(data_dir, dataset, DATASETS_DETAILS[dataset]['query']))
        ref = anndata.read_h5ad(os.path.join(data_dir, dataset, DATASETS_DETAILS[dataset]['reference']))
        batch_key = DATASETS_DETAILS[dataset]['batch_key']
        celltype_key = DATASETS_DETAILS[dataset]['cell_type_key']
        # Convert numeric predictions to label names
        label_names = [id_maps[i] for i in labels]
        fed_names = [id_maps[i] for i in fed]
        cent_names = [id_maps[i] for i in cent]

        label_set = sorted(set(label_names), key=str)
        extra_labels = sorted(set(fed_names + cent_names) - set(label_set), key=str)
        unique_celltypes = label_set + extra_labels

        color_mapping = generate_palette(unique_celltypes)

        plot_confusion_with_deltas(
            dataset, cent, fed, labels, id_maps,
            color_mapping,
            os.path.join(ANNOTATION_PLOTS_DIR, 'confusion_matrix', f"{filename}.png")
        )

        # query.obs['split'] = "query"
        # query.obs['prediction_federated'] = fed_names
        # query.obs['prediction_centralized'] = cent_names
        # ref.obs['split'] = "reference"
        # ref.obs['prediction_federated'] = None
        # ref.obs['prediction_centralized'] = None
        # adata = anndata.concat([query, ref], label='split_fake')
        # ref_only_celltypes = sorted(set(ref.obs[celltype_key].values) - set(unique_celltypes))
        # all_colors = generate_palette(unique_celltypes + ref_only_celltypes)
        # all_colors.update(generate_palette(sorted(set(adata.obs[batch_key].values))))
        #
        # # Compute UMAP once
        # if 'X_umap' not in adata.obsm:
        #     raise ValueError
        # umap_dir = f"{ANNOTATION_PLOTS_DIR}/UMAPs/{dataset}"
        # sc.settings.figdir = umap_dir
        # plot_entire_ds_umap(adata, all_colors, batch_key, celltype_key, dataset)
        # plot_reference_umap(adata, all_colors, batch_key, celltype_key, dataset)
        # plot_query_pre_umap(adata, celltype_key, color_mapping, dataset, filename)




def plot_batch_effect(dataset, data_dir):
    query = anndata.read_h5ad(os.path.join(data_dir, dataset, DATASETS_DETAILS[dataset]['query']))
    ref = anndata.read_h5ad(os.path.join(data_dir, dataset, DATASETS_DETAILS[dataset]['reference']))
    batch_key = DATASETS_DETAILS[dataset]['batch_key']
    celltype_key = DATASETS_DETAILS[dataset]['cell_type_key']
    query.obs['split'] = "query"
    ref.obs['split'] = "reference"
    adata = anndata.concat([query, ref], label='split_fake')
    top10 = adata.obs[celltype_key].value_counts().nlargest(10).index
    adata = adata[adata.obs[celltype_key].isin(top10)].copy()
    unique_celltypes = sorted(set(adata.obs[celltype_key].values), key=str)
    color_mapping = generate_palette(unique_celltypes)
    batch_color_map = generate_palette(sorted(set(adata.obs[batch_key].values)))
    color_mapping.update(batch_color_map)

    # Compute UMAP once
    if 'X_umap' not in adata.obsm:
        raise ValueError
    umap_dir = f"{ANNOTATION_PLOTS_DIR}/UMAPs/batch-effect-{dataset}"
    sc.settings.figdir = umap_dir
    plot_entire_ds_umap(adata, color_mapping, batch_key, celltype_key, dataset)

def plot_query_pre_umap(adata, celltype_key, color_mapping, dataset, filename):
    # Query
    plot_umap_and_legend(adata[adata.obs['split'] == 'query'], color_mapping,
                         color=[celltype_key, "prediction_centralized", "prediction_federated"],
                         title=["Query: Ground Truth", "Centralized", "Federated"], filename=f"_{dataset}_query.png")
    plot_umap_and_legend(adata[adata.obs['split'] == 'query'], color_mapping,
                         color=[celltype_key],
                         title=["Query: Ground Truth"],
                         filename=f"_{dataset}_query_celltype.png")
    plot_umap_and_legend(adata[adata.obs['split'] == 'query'], color_mapping,
                         color=["prediction_centralized"],
                         title=["Centralized"],
                         filename=f"_{dataset}_query_centralized.png")
    plot_umap_and_legend(adata[adata.obs['split'] == 'query'], color_mapping,
                         color=["prediction_federated"],
                         title=["Federated"],
                         filename=f"_{dataset}_query_federated_{filename}.png")

def plot_best_fed_vs_centralized(df, subdir='efficiency'):
    sns.set(style='whitegrid')
    metrics = df['Metric'].unique()
    aggregations = {
        "weighted-FedAvg": "FedAvg",
        "SMPC-weighted-FedAvg": "FedAvg-SMPC",
        "SMPC-weighted-FedProx": "FedProx-SMPC",
        "weighted-FedProx": "FedProx"
    }
    palette = generate_palette(aggregations.values())
    df.sort_values("Dataset", ascending=False, inplace=True)
    n_datasets = len(df['Dataset'].unique())
    for metric in metrics:
        metric_df = df[df['Metric'] == metric]

        # Centralized perf at n_epochs == 20
        central_df = metric_df[(metric_df['n_epochs'] == 20) & (metric_df['Aggregation'] == "centralized")]
        central_df.sort_values("Dataset", ascending=False, inplace=True)
        assert len(central_df) == n_datasets, f"Expected {n_datasets} datasets for centralized, got {len(central_df)}"
        centralized = dict(zip(central_df['Dataset'], central_df['Value']))

        # Best federated
        fed_df = metric_df[metric_df['Aggregation'].isin(aggregations.keys())].copy()
        fed_df['Aggregation'] = fed_df['Aggregation'].map(aggregations)
        best_df = (
            fed_df.groupby(['Dataset', 'Aggregation'])['Value']
            .max()
            .reset_index()
        )
        best_df.sort_values("Dataset", ascending=False, inplace=True)
        # Plot
        plt.figure(figsize=(7, 4))
        ax = sns.barplot(data=best_df, x='Dataset', y='Value', hue='Aggregation', dodge=True, palette=palette)
        group_width = 0.8  # seaborn default
        for i, (dataset, value) in enumerate(centralized.items()):
            center = i  # x-axis location of dataset group
            left = center - group_width / 2
            right = center + group_width / 2
            ax.hlines(y=value, xmin=left, xmax=right, colors='black', linestyles='--', linewidth=1)
            ax.text(center, value + 0.01, f'{value:.2f}', ha='center', fontsize=12  , color='black')

        ax.set_ylabel(metric, fontsize=16)
        ax.set_xlabel('Dataset', fontsize=16)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=14)
        ax.set_ylim(0.2, 1.05)
        ax.legend_.remove()
        plt.tight_layout()
        plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{subdir}/best_fed_vs_central_{metric}.png", dpi=300)
        plt.close()
        legend_elements = [Patch(facecolor=palette[agg], label=agg) for agg in aggregations.values()]

        fig, ax = plt.subplots(figsize=(6, 1))
        ax.axis('off')
        legend = ax.legend(handles=legend_elements, loc='center', ncol=len(legend_elements), frameon=False, fontsize=20)
        plt.tight_layout()
        plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{subdir}/federated_methods_legend.png", dpi=300, bbox_inches='tight',
                    transparent=True)
        plt.close()

def plot_best_fed_myeloid_vs_centralized(df, subdir='efficiency'):
    sns.set(style='whitegrid')
    os.makedirs(f"{ANNOTATION_PLOTS_DIR}/{subdir}", exist_ok=True)

    target_agg = "SMPC-weighted-FedProx"
    cent_dataset = 'myeloid-top30'
    methods = ['Top5', 'Top10', 'Top20', 'Top30']
    metrics = df['Metric'].unique()
    plot_data = []
    legend_labels = {}

    for ds in methods:
        best_row_idx = df[
            (df['Dataset'] == ds) &
            (df['Aggregation'] == target_agg) &
            (df['Metric'] == 'Macro-F1') &
            (df['n_epochs'] == 1) &
            (df['mu'] == 0.01)
        ].Value.idxmax()
        best_row = df.loc[best_row_idx]
        best_round = best_row['Round']
        best_epoch = 1
        best_mu = 0.01
        print(ds, best_round, best_epoch, best_mu)
        legend_labels[ds] = f"{ds} ({int(best_round)} rounds)"

        for metric in metrics:
            val = df[
                (df['Dataset'] == ds) &
                (df['Aggregation'] == target_agg) &
                (df['Round'] == best_round) &
                (df['n_epochs'] == best_epoch) &
                (df['mu'] == best_mu) &
                (df['Metric'] == metric)
            ]['Value']
            assert len(val) == 1, f"Expected one value for {ds} {metric}, got {len(val)}"
            val = val.values[0]
            plot_data.append({
                'Metric': metric,
                'Method': ds,
                'Value': val
            })

    # Centralized
    central_row = df[
        (df['Dataset'] == 'Top30') &
        (df['Aggregation'] == 'centralized') &
        (df['n_epochs'] == 20)
    ]
    assert len(central_row) == len(metrics), f"Expected {len(metrics)} metrics for centralized, got {len(central_row)}"

    for metric in metrics:
        val = central_row[central_row['Metric'] == metric]['Value'].values[0]
        plot_data.append({
            'Metric': metric,
            'Method': 'centralized',
            'Value': val
        })

    legend_labels['centralized'] = 'Centralized'

    plot_df = pd.DataFrame(plot_data)

    plot_df['Method'] = pd.Categorical(
        plot_df['Method'],
        categories=['centralized'] + methods,
        ordered=True
    )

    palette = generate_palette(plot_df['Method'].unique())
    palette['centralized'] = 'black'  # Ensure centralized is black

    # --- Bar Plot (No Legend) ---
    plt.figure(figsize=(10, 4))
    ax = sns.barplot(data=plot_df, x='Metric', y='Value', hue='Method', palette=palette)
    ax.set_ylabel("Score", fontsize=20)
    ax.set_xlabel("")
    ax.set_ylim(0, 0.7)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=18)
    ax.legend_.remove()
    plt.tight_layout()
    plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{subdir}/grouped_fedprox_smpc_vs_central.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Legends ---
    handles = [Patch(facecolor=palette[m], label=legend_labels[m]) for m in plot_df['Method'].cat.categories]

    # Horizontal Legend
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.axis('off')
    ax.legend(handles=handles, loc='center', ncol=len(handles), frameon=False,
              fontsize=30, handletextpad=0.4, columnspacing=0.8)
    plt.tight_layout()
    plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{subdir}/legend_horizontal.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # Vertical Legend
    fig, ax = plt.subplots(figsize=(4, len(handles) * 0.5))
    ax.axis('off')
    ax.legend(handles=handles, loc='center', ncol=1, frameon=False, fontsize=11, title="Method (Best Params)", title_fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{subdir}/legend_vertical.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


def plot_myeloid_over_rounds_vs_central(df, subdir='myeloid'):
    sns.set(style='whitegrid')
    os.makedirs(f"{ANNOTATION_PLOTS_DIR}/{subdir}", exist_ok=True)

    target_agg = "SMPC-weighted-FedProx"
    selected_datasets = ['Top5', 'Top10', 'Top20', 'Top30']
    palette = generate_palette(selected_datasets)

    # Use linestyles instead of markers
    linestyles = {'Top5': 'dashdot', 'Top10': 'dotted', 'Top20': 'dashed', 'Top30': 'solid'}
    metrics = df['Metric'].unique()

    for metric in metrics:
        plt.figure(figsize=(6, 4))
        found_any = False

        for ds in selected_datasets:
            sub_df = df[
                (df['Dataset'] == ds) &
                (df['Aggregation'] == target_agg) &
                (df['n_epochs'] == 1) &
                (df['mu'] == 0.01) &
                (df['Metric'] == metric)
            ].sort_values("Round")

            if sub_df.empty:
                continue

            found_any = True
            plt.plot(sub_df['Round'], sub_df['Value'],
                     label=ds,
                     color=palette[ds],
                     linestyle=linestyles.get(ds, 'solid'),
                     linewidth=2)

        if not found_any:
            raise ValueError(f"No data found for metric '{metric}' across datasets")

        # Centralized horizontal line
        central_val = df[
            (df['Dataset'] == 'Top30') &
            (df['Aggregation'] == 'centralized') &
            (df['n_epochs'] == 20) &
            (df['Metric'] == metric)
        ]['Value'].values[0]

        plt.axhline(y=central_val, linestyle='solid', color='black', linewidth=2, label='Centralized')

        plt.xlabel("Round", fontsize=22)
        plt.ylabel(metric, fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=16)
        plt.ylim(0, 0.7)
        plt.xlim(0, 200)
        plt.tight_layout()

        out_path = f"{ANNOTATION_PLOTS_DIR}/{subdir}/fedprox_smpc_line_{metric}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()


def plot_myeloid_over_rounds(df, subdir='myeloid'):
    sns.set(style='whitegrid')
    os.makedirs(f"{ANNOTATION_PLOTS_DIR}/{subdir}", exist_ok=True)

    target_agg = "SMPC-weighted-FedProx"
    selected_datasets = ['Top5', 'Top10', 'Top20', 'Top30']
    palette = generate_palette(selected_datasets)
    linestyles = {'Top5': 'dashdot', 'Top10': 'dotted', 'Top20': 'dashed', 'Top30': 'solid'}

    metrics = df['Metric'].unique()
    n_metrics = len(metrics)

    fig, ax = plt.subplots(n_metrics, 1, figsize=(4, 2 * n_metrics), sharex=True)

    for i, metric in enumerate(metrics):
        ax_i = ax[i]
        found_any = False

        for ds in selected_datasets:
            sub_df = df[
                (df['Dataset'] == ds) &
                (df['Aggregation'] == target_agg) &
                (df['n_epochs'] == 1) &
                (df['mu'] == 0.01) &
                (df['Metric'] == metric)
            ].sort_values("Round")

            if sub_df.empty:
                continue

            found_any = True
            ax_i.plot(sub_df['Round'], sub_df['Value'],
                      color=palette[ds],
                      linestyle=linestyles.get(ds, 'solid'),
                      linewidth=2)

        if not found_any:
            raise ValueError(f"No data found for metric '{metric}' across datasets")

        # Centralized line for Top30
        central_val = df[
            (df['Dataset'] == 'Top30') &
            (df['Aggregation'] == 'centralized') &
            (df['n_epochs'] == 20) &
            (df['Metric'] == metric)
        ]['Value'].values[0]
        ax_i.axhline(y=central_val, linestyle='solid', color='black', linewidth=2)

        # Y-axis styling
        ax_i.set_ylabel(metric, fontsize=16)
        ax_i.set_ylim(0, 0.69)
        ax_i.tick_params(axis='y', labelsize=12)

        ax_i.set_xlim(-1, 200)
        ax_i.set_xticks(range(0, 201, 50))
        ax_i.tick_params(axis='x', labelbottom=True, bottom=True, labelsize=0 if i < n_metrics - 1 else 18)

    plt.subplots_adjust(hspace=0.02)  # Tighter vertical spacing
    plt.tight_layout(h_pad=0.03)
    plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{subdir}/myeloid_over_rounds.png", dpi=300)
    plt.close()




def plot_reference_umap(adata, all_colors, batch_key, celltype_key, dataset):
    ref_adata = adata[adata.obs['split'] == 'reference']

    plot_umap_and_legend(
        ref_adata,
        all_colors,
        color=[celltype_key, batch_key],
        title=["Reference: Cell Type", "Reference: Batch"],
        filename=f"_{dataset}_reference.png"
    )

    plot_umap_and_legend(
        ref_adata,
        all_colors,
        color=[celltype_key],
        title=["Reference: Cell Type"],
        filename=f"_{dataset}_reference_celltype.png"
    )

    if dataset in ['myeloid-top5', 'myeloid-top10', 'myeloid-top20', 'myeloid-top30']:
        plot_batch_umap_with_legend(
            ref_adata,
            color_key=batch_key,
            title="Reference: Batch",
            umap_filename=f"{sc.settings.figdir}/_{dataset}_reference_batch_colorbar.png",
            legend_filename=f"{sc.settings.figdir}/_{dataset}_reference_batch_legend.png",
            colormap='turbo'  # or 'nipy_spectral', 'tab20', etc.
        )
    else:
        plot_umap_and_legend(
            ref_adata,
            all_colors,
            color=[batch_key],
            title=["Reference: Batch"],
            filename=f"_{dataset}_reference_batch.png"
        )


def plot_entire_ds_umap(adata, all_colors, batch_key, celltype_key, dataset):
    # Entir datasets
    plot_umap_and_legend(adata, all_colors, [celltype_key, batch_key],
                         title=["Cell Type", "Batch"], filename=f"_{dataset}.png")
    plot_umap_and_legend(adata, all_colors, [celltype_key],
                         title=["Cell Type"], filename=f"_{dataset}_celltype.png")
    plot_umap_and_legend(adata, all_colors, [batch_key],
                         title=["Batch"], filename=f"_{dataset}_batch.png")


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scanpy as sc
import numpy as np

def plot_batch_umap_with_legend(
        adata,
        color_key='batch',
        title='UMAP by Batch',
        umap_filename='umap_batches.png',
        legend_filename='umap_batches_legend.png',
        colormap='turbo'
):
    # Step 1: Prepare batch labels
    adata.obs[color_key] = adata.obs[color_key].astype(str)
    unique_batches = adata.obs[color_key].unique().tolist()
    print(f"num batches: {len(unique_batches)}")
    print(f"unique batches: {unique_batches}")
    # unique_batches = sorted(unique_batches, key=lambda x: int(x) if x.isdigit() else len(unique_batches))

    # Replace 'rest' with len(unique_batches)
    if 'rest' in unique_batches:
        unique_batches.append(str(len(unique_batches)))
        unique_batches.remove('rest')
        adata.obs[color_key] = adata.obs[color_key].replace('rest', str(len(unique_batches)))
    unique_batches = sorted(unique_batches, key=lambda x: int(x))
    # unique_batches = [int(b) for b in unique_batches]
    print(unique_batches)
    # Step 2: Map batch labels to integers for coloring
    batch_to_index = {b: i for i, b in enumerate(unique_batches)}
    # adata.obs['color_idx'] = adata.obs[color_key].astype(int)
    adata.obs['color_idx'] = adata.obs[color_key].map(batch_to_index).astype(int)
    # Step 3: Generate discrete colors using a colormap
    cmap = cm.get_cmap(colormap, len(unique_batches))
    color_list = [cmap(i) for i in range(len(unique_batches))]

    # Step 4: Plot UMAP using Scanpy with fixed colors
    # color_palette = {str(b): color_list[b-1] for b in unique_batches}
    color_palette = {b: color_list[i] for i, b in enumerate(unique_batches)}
    print(color_palette)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc.pl.umap(
        adata,
        color=color_key,
        palette=color_palette,
        show=False,
        ax=ax,
        frameon=False,
        title=title
    )
    plt.tight_layout()
    plt.savefig(umap_filename, dpi=300)
    plt.close()

    # Step 5: Plot standalone circle legend
    fig_legend, ax_legend = plt.subplots(figsize=(3, 12))
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=str(b),
                   markerfacecolor=color_palette[b], markersize=8)
        for b in unique_batches
    ]
    ax_legend.legend(
        handles,
        unique_batches,
        loc='center',
        fontsize='small',
        frameon=False,
        handletextpad=0.5,  # space between marker and text
        borderaxespad=0.2,  # space around the legend box
        labelspacing=0.05,  # space between legend entries (rows)
        handlelength=1.5,  # length of the legend markers
        ncol=1
    )

    ax_legend.axis('off')
    plt.tight_layout()
    plt.savefig(legend_filename, dpi=300)
    plt.close()



def plot_batch_legend_only(adata, color_key='batch', colormap='turbo', filename='batch_legend.png'):
    unique_batches = sorted(adata.obs[color_key].unique())
    batch_to_index = {b: i for i, b in enumerate(unique_batches)}
    cmap = cm.get_cmap(colormap, len(unique_batches))

    colors = [cmap(batch_to_index[b]) for b in unique_batches]

    fig_legend, ax_legend = plt.subplots(figsize=(3, 12))
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(b), markerfacecolor=c, markersize=8) for b, c in
               zip(unique_batches, colors)]
    ax_legend.legend(handles, unique_batches, loc='center', fontsize='small', frameon=False)
    ax_legend.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_umap_and_legend(adata, color_mapping, color, title, filename):
    # 1. UMAP for Reference Subset
    sc.pl.umap(
        adata,
        color=color,
        title=title,
        palette=color_mapping,
        frameon=False,
        ncols=len(color),
        legend_loc=None,  # <- disable inline legend
        show=False,
        save=filename
    )
    plt.close('all')
    for c, t in zip(color, title):
        umap_legend_plot(adata, c, color_mapping , filename=f"{sc.settings.figdir}/{filename.replace('.png', f'-{t}-legend.png')}")
        plt.close('all')


def umap_legend_plot(adata, color, color_mapping, filename):
    fig = plt.figure(figsize=(15, 5))
    sc.pl.umap(adata, color=color, palette=color_mapping, show=False)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.close()
    unique_celltypes = sorted(set(adata.obs[color].values))
    sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: unique_celltypes.index(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)

    fig_legend, ax_legend = plt.subplots(figsize=(3, 9))  # Separate figure for the legend
    # ax_legend.legend(sorted_handles, sorted_labels, loc='center', fontsize='small', frameon=False, ncol=1)
    ax_legend.legend(
        sorted_handles,
        sorted_labels,
        loc='center',
        fontsize='small',
        frameon=False,
        handletextpad=0.5,  # space between marker and text
        borderaxespad=0.2,  # space around the legend box
        labelspacing=0.05,  # space between legend entries (rows)
        handlelength=1.5,  # length of the legend markers
        ncol=1
    )
    ax_legend.axis('off')
    plt.savefig(filename, dpi=300)
    plt.close()


# def plot_umap_and_conf_matrix(dataset, adata, id_maps, labels, cent, fed):
#         # Convert IDs to string labels for plotting
#         label_names = [id_maps[i] for i in labels]
#         fed_names = [id_maps[i] for i in fed]
#         cent_names = [id_maps[i] for i in cent]
#         # unique_celltypes = sorted(set(label_names + fed_names + cent_names), key=lambda x: str(x))
#         label_set = sorted(set(label_names), key=lambda x: str(x))
#         extra_labels = sorted(set(fed_names + cent_names) - set(label_set), key=lambda x: str(x))
#         unique_celltypes = label_set + extra_labels
#
#         color_mapping = generate_palette(unique_celltypes)
#
#
#         # Plot UMAP
#         plot_umaps(
#             adata, cent_names, fed_names, label_names, unique_celltypes,
#             f"{dataset}_umap.png",
#             f"{dataset}_legend.png",
#             color_mapping,
#             plot_legend=True
#         )
#         plot_umaps(
#             adata, cent_names, fed_names, label_names, unique_celltypes,
#             f"{dataset}_umap.png",
#             f"{dataset}_legend.png",
#             color_mapping,
#             plot_legend=False
#         )
#
#         # Plot Confusion Matrix
#         plot_confusion_with_deltas(
#             dataset, cent, fed, labels, id_maps,
#             color_mapping,
#             os.path.join(ANNOTATION_PLOTS_DIR, f"{dataset}_confusion_delta.svg")
#         )


# def plot_umap_and_conf_matrix(root_dir, data_dir, res_pkl_file, res_df_file):
#     df = pd.read_csv(res_df_file)
#     best_fed = {ds: df.loc[df[(df.Dataset==ds) & (df.Metric == 'Accuracy')]["Value"].idxmax()] for ds in df.Dataset.unique()}
#     results = load_results_pkl(root_dir, res_pkl_file, best_fed)
#
#     for dataset in results.keys():
#         adata = load_query_datasets(data_dir, dataset)
#         predictions_centralized = results[dataset]['centralized']['predictions']
#         predictions_federated = results[dataset]['federated']['predictions']
#         labels = results[dataset]['centralized']['labels']  # Assuming the labels are the same for all modes
#         unique_celltypes = results[dataset]['centralized']['unique_celltypes']
#         assert results[dataset]['federated']['id_maps'] == results[dataset]['centralized']['id_maps']
#         id_maps = results[dataset]['federated']['id_maps']
#         labels = [id_maps[c] for c in labels]
#         predictions_federated = [id_maps[c] for c in predictions_federated]
#         predictions_centralized = [id_maps[c] for c in predictions_centralized]
#         palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#         palette_ = palette_ * 3  # Extend the palette if needed
#         color_mapping = {c: palette_[i] for i, c in enumerate(unique_celltypes)}
#         color_mapping = plot_confusion_matrices(dataset, results, color_mapping)
#
#         plot_umaps(adata, predictions_centralized, predictions_federated, labels, unique_celltypes,
#                    f"{dataset}_umap_plots.png", f"{dataset}_legend.png", color_mapping)
#         plot_umaps(adata, predictions_centralized, predictions_federated, labels, unique_celltypes,
#                    f"{dataset}_umap_plots.png", f"{dataset}_legend.png", color_mapping, plot_legend=True)


def plot_umaps(adata, predictions_centralized, predictions_federated, labels, unique_celltypes, file_name, legend_file_name, color_mapping, plot_legend=False):
    if 'X_umap' not in adata.obsm.keys():
        print(f"X_umap not found in adata.obsm {file_name} ==> the keys are: {adata.obsm.keys()}")
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')
        sc.tl.umap(adata)
    adata.obs['cell_type'] = labels
    if not plot_legend:
        fig, axes = plt.subplots(3, 1, figsize=(4, 12))
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


# def embedding_boxplot(data_dir, datasets, plots_dir, img_format='svg'):
#
#     metrics = ['accuracy', 'precision', 'recall', 'macro_f1']
#
#     # Initialize an empty DataFrame to hold all results
#     df = pd.DataFrame(columns=['Dataset', 'Type', 'Metric', 'Value'])
#
#     fedscgpt_file_path = {ds: f"{data_dir}/{ds}/federated/evaluation_metrics.csv" for ds in datasets}
#     fedscgpt_smpc_file_path = {ds: f"{data_dir}/{ds}/federated/smpc/evaluation_metrics.csv" for ds in datasets}
#     scgpt_file_path = {ds: f"{data_dir}/{ds}/centralized/evaluation_metrics.csv" for ds in datasets}
#
#     for ds in datasets:
#         adata_path = Path(data_dir).parents[1]/'data/scgpt/benchmark'/ds/datasets_details[ds]['h5ad_file'].split("|")[0]
#         ref = anndata.read_h5ad(adata_path)
#         batches = list(sorted(ref.obs[datasets_details[ds]['batch_key']].unique()))
#         # Load centralized and federated results
#         scgpt = pd.read_csv(scgpt_file_path[ds])
#         fedscgpt = pd.read_csv(fedscgpt_file_path[ds])
#         fedscgpt_smpc = pd.read_csv(fedscgpt_smpc_file_path[ds])
#         rows = []
#         # Append centralized and federated results to the DataFrame
#         for metric in metrics:
#             rows.append({
#                 'Dataset': ds,
#                 'Type': 'Centralized',
#                 'Metric': metric,
#                 'Value': scgpt[metric].values[0]
#             })
#             rows.append({
#                 'Dataset': ds,
#                 'Type': 'Federated',
#                 'Metric': metric,
#                 'Value': fedscgpt[metric].values[0]
#             })
#             rows.append({
#                 'Dataset': ds,
#                 'Type': 'Federated-SMPC',
#                 'Metric': metric,
#                 'Value': fedscgpt_smpc[metric].values[0]
#             })
#         df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
#         # Collect and append client results
#         client_dir_path = os.path.join(data_dir, ds, "centralized")
#         rows = []
#         for client_dir in os.listdir(client_dir_path):
#             if client_dir.startswith("client"):
#                 client_metrics = pd.read_csv(os.path.join(client_dir_path, client_dir, "evaluation_metrics.csv"))
#                 client_num = int(os.path.basename(client_dir).split("_")[1])
#                 client_batch_value = batches[client_num]
#                 for metric in metrics:
#                     rows.append({
#                         'Dataset': ds,
#                         'Type': 'Client',
#                         'Metric': metric,
#                         'Value': client_metrics[metric].values[0],
#                         'Batch': client_batch_value
#                     })
#         df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
#     # display_federated_performance_report(df)
#     per_metric_annotated_scatterplot(df, plots_dir, img_format)

def embedding_boxplot(df_all_metrics, datasets, subdir, img_format='svg'):
    metrics = ['Accuracy', 'Precision', 'Recall', 'Macro-F1']
    output_df = pd.DataFrame(columns=['Dataset', 'Type', 'Metric', 'Value', 'Batch'])

    for ds in datasets:
        # Filter rows for this dataset
        df_ds = df_all_metrics[df_all_metrics['Dataset'] == ds]

        for _, row in df_ds.iterrows():
            aggregation = row['Aggregation']
            if aggregation == "centralized":
                agg_type = "Centralized"
            elif aggregation == "federated":
                agg_type = "Federated"
            elif "smpc" in row['Source']:
                agg_type = "Federated-SMPC"
            elif aggregation.startswith("client_"):
                agg_type = "Client"
            else:
                agg_type = "Unknown"

            for metric in metrics:
                new_row = {
                    "Dataset": ds,
                    "Type": agg_type,
                    "Aggregation": row['Aggregation'],
                    "Metric": metric,
                    "Value": row[metric],
                    "Batch": None  # optional
                }

                # Add batch name as 'client_X' for client results
                if aggregation.startswith("client_"):
                    new_row["Batch"] = aggregation

                output_df = pd.concat([output_df, pd.DataFrame([new_row])], ignore_index=True)
    if subdir == 'covid':
        figsize = (7, 5)
    else:
        figsize = (5, 5)
    ylim = 0.7 if subdir == 'myeloid' else 1
    per_metric_ref_map_scatterplot(output_df, f"{EMBEDDING_PLOTS_DIR}/{subdir}", img_format, figsize, ylim=ylim)


def per_metric_ref_map_scatterplot(df, plots_dir, img_format='png', figsize=(7,5), x_spacing=0.8, ylim=1):
    """
    Plot annotated scatterplots per metric, with adjustable horizontal spacing.

    Parameters:
    - df: DataFrame with ['Dataset', 'Type', 'Metric', 'Value', 'Aggregation', 'Batch']
    - plots_dir: output directory for plots
    - img_format: format for saved plots
    - proximity_threshold: (not used in this version, retained for compatibility)
    - x_spacing: float, space between datasets on x-axis (default: 1.0)
    """
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    datasets = df['Dataset'].unique()
    metrics = df['Metric'].unique()

    client_mask = df['Type'].str.lower() == 'client'
    cmap = safe_extended_palette

    # Aggregation marker, label, and x-offset
    agg_map = {
        'federated': (FEDSCGPT_MARKER, 'Federated', -0.15),
        'smpc': (FEDSCGPT_SMPC_MARKER, 'Federated-SMPC', 0.15),
    }

    for metric in metrics:
        plt.figure(figsize=figsize)

        for i, ds in enumerate(datasets):
            x_base = i * x_spacing
            df_ds = df[(df['Dataset'] == ds) & (df['Metric'] == metric)]

            # --- Plot client points with jitter ---
            clients = df_ds[df_ds['Type'].str.lower() == 'client']
            if not clients.empty:
                vals = clients['Value'].values
                jitter = np.random.uniform(-0.05, 0.05, size=len(vals))
                x_vals = x_base + jitter
                plt.scatter(x_vals, vals,
                            color=cmap[i % len(cmap)],
                            edgecolor='black',
                            s=50, alpha=0.7,
                            label="Client" if i == 0 else None)

            # --- Centralized as horizontal dashed line ---
            centralized = df_ds[df_ds['Type'] == 'Centralized']
            if not centralized.empty:
                val = centralized['Value'].values[0]
                plt.hlines(val, x_base - 0.3, x_base + 0.3,
                           color=cmap[i % len(cmap)],
                           linestyle='--', linewidth=2,
                           label="Centralized" if i == 0 else None)

            # --- Plot federated variants with marker + x-offset ---
            for agg_key, (marker, label_text, offset) in agg_map.items():
                sub = df_ds[df_ds['Aggregation'].str.lower() == agg_key]
                if not sub.empty:
                    val = sub['Value'].values[0]
                    x_pos = x_base + offset
                    color = cmap[i % len(cmap)]
                    fill = color if 'smpc' in agg_key else 'white'

                    plt.scatter(x_pos, val,
                                marker=marker,
                                s=100,
                                facecolors=fill,
                                edgecolors='black',
                                zorder=5,
                                label=label_text if i == 0 else None)

        # --- Plot settings ---
        plt.xlabel('')
        plt.ylim(0, ylim)
        plt.ylabel(metric, fontsize=24)
        plt.xticks(
            ticks=[i * x_spacing for i in range(len(datasets))],
            labels=[DS_NAME_MAP.get(d, d) for d in datasets],
            fontsize=22
        )
        plt.yticks(fontsize=18)
        plt.tight_layout()

        out_path = os.path.join(plots_dir, f"{metric}_scatterplot_annotated.{img_format}")
        plt.savefig(out_path, dpi=300, format=img_format, bbox_inches='tight')
        plt.close()
        # Regenerate external legend
        plot_legend_embedding(plots_dir, img_format)




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
    if ds_name in ["covid", "covid-corrected", "covid-fed-corrected"]:
        return "batch_group"
    if ds_name == "lung":
        return "sample"
    if ds_name == "hp":
        return "batch"
    if ds_name == "ms":
        return "split_label"
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
        'client_premotor cortex': 'Premotor',
        'Northwestern_Misharin_2018Reyfman': 'Misharin',
    }
    return replace.get(batch_value, batch_value)



def shorten_celltype_value(celltype):
    replace = {
        "CD4+ T cells": "CD4 T",
        "CD8+ T cells": "CD8 T",
        "CD14+ Monocytes": "CD14 Mono",
        "CD16+ Monocytes": "CD16 Mono",
        "CD20+ B cells": "CD20 B",
        "Dendritic cell": "Dendritic",
        "Proliferating Macrophage": "Prolif Macro",
        "Megakaryocyte progenitors": "Megakaryo prog",
        "Signaling Alveolar Epithelial Type 2": "Sig AET2",  # if present
        "M2 Macrophage": "M2 Macro",
        "IGSF21+ Dendritic": "IGSF21 DC",
        "Mast cells": "Mast",
        "NKT cells": "NKT",
    }
    return replace.get(celltype, celltype)


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
        'Centralized':           {'kind':'line', 'linestyle':'--', 'linewidth':2},
        'Federated':        {'marker':FEDSCGPT_MARKER, 's':100, 'edgecolor':'black'},
        'Federated-SMPC':   {'marker':FEDSCGPT_SMPC_MARKER, 's':100, 'edgecolor':'black'}
    }
    default_marker = 'X'

    for metric in metrics:
        plt.figure(figsize=(8,5))
        # --- Plot client points with jitter + labels ---
        cmap = safe_extended_palette
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
                # label = shorten_batch_value(b)
                label = batch_map.get(ds, {}).get(b,b)
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
                   [DS_NAME_MAP.get(d, d) for d in datasets],
                   fontsize=18)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"{metric}_scatterplot_annotated.{img_format}")
        plt.savefig(out_path, dpi=300, format=img_format, bbox_inches='tight')
        plt.close()

        # regenerate legend (assumed your helper handles all markers/lines)
        plot_legend_embedding(plots_dir, img_format)

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


def plot_legend_embedding(plots_dir, img_format='svg', layout='row'):
    """
    Plot a separate figure containing only the legend for:
      – scGPT (centralized)
      – FedscGPT (federated)
      – FedscGPT-SMPC (federated + SMPC)
      – Clients

    Args:
        plots_dir (str): Directory to save the legend image.
        img_format (str): File format (e.g., 'svg', 'png').
        layout (str): 'row' for horizontal, 'column' for vertical layout.
    """
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Centralized'),
        Line2D([0], [0], marker=FEDSCGPT_MARKER, color='w', markersize=10,
               markeredgecolor='black', label='Federated'),
        Line2D([0], [0], marker=FEDSCGPT_SMPC_MARKER, color='w', markersize=10,
               markeredgecolor='black', label='Federated-SMPC'),
        Line2D([0], [0], marker='o', color='w', markersize=8,
               markeredgecolor='black', label='Clients'),
    ]

    if layout == 'row':
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.legend(
            handles=legend_elements,
            loc='center',
            ncol=len(legend_elements),
            fontsize=12,
            frameon=False,
            handletextpad=0.4,
            columnspacing=0.8,
        )
    elif layout == 'column':
        fig, ax = plt.subplots(figsize=(2.5, len(legend_elements)))
        ax.legend(
            handles=legend_elements,
            loc='center',
            ncol=1,
            fontsize=12,
            frameon=False,
            handletextpad=0.4
        )
    else:
        raise ValueError("layout must be either 'row' or 'column'")

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/legend.{img_format}", format=img_format, dpi=300, bbox_inches='tight', transparent=True)
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

# def accuracy_annotated_scatterplot(df, plots_dir, img_format='svg', proximity_threshold=0.2):
#     """
#     Plot data using Matplotlib from a pandas DataFrame, with each scatter point annotated by its corresponding 'Batch' value.
#     Adjusts the text to the left or right dynamically to avoid overlap for points with close y-values.
#
#     Parameters:
#     - df: DataFrame containing the data to plot.
#     - plots_dir: Directory to save the plots.
#     - img_format: Format to save the images (e.g., 'svg').
#     - proximity_threshold: Defines the closeness of y-values to consider them overlapping (default = 0.1).
#     - legend_inside: Boolean flag indicating whether to place the legend inside the figure (default = False).
#     """
#     if not os.path.exists(plots_dir):
#         os.makedirs(plots_dir)
#
#     metrics = df['Metric'].unique()
#     datasets = df['Dataset'].unique()
#     print(datasets)
#
#     for metric in metrics:
#         plt.figure(figsize=(5, 5))
#
#         # Separate client data and centralized/federated data
#         df_metric = df[df['Metric'] == metric]
#         scgpt = df_metric[df_metric['Type'] == 'scGPT']
#         fedscgpt_smpc = df_metric[df_metric['Type'] == 'FedscGPT-SMPC']
#         fedscgpt = df_metric[df_metric['Type'] == 'FedscGPT']
#         client_data = df_metric[~df_metric['Type'].isin(['scGPT', 'FedscGPT-SMPC', 'FedscGPT'])]
#
#         # Scatter plot for client data
#         colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgrey', 'lightyellow']  # Extend as needed
#         scatter_plots = []
#         for i, dataset in enumerate(datasets):
#             dataset_clients = client_data[client_data['Dataset'] == dataset]
#             client_values = dataset_clients['Value'].values
#             client_batches = dataset_clients['Type'].values
#             # Scatter each client point with a slight horizontal offset to avoid overlap
#             jitter = 0.05  # Add some horizontal jitter to avoid overlap
#             x_jitter = np.random.uniform(-jitter, jitter, size=client_values.shape)
#             scatter = plt.scatter([i + 1 + x for x in x_jitter], client_values,
#                                   color=colors[i % len(colors)], edgecolor='black', s=50, alpha=0.7,
#                                   label=f"Client Data - {dataset}")
#             scatter_plots.append(scatter)
#
#             # Determine proximity of y-values to decide label positions
#             for j, (x, y, batch) in enumerate(zip([i + 1 + x for x in x_jitter], client_values, client_batches)):
#                 batch_label = shorten_batch_value(batch)
#
#                 # Check if other points are "close enough" in y-value using the proximity threshold
#                 close_points = np.sum(np.abs(client_values - y) < proximity_threshold)
#                 annotation_font_size = 12
#                 if close_points > 1:  # If there are other points within the threshold range
#                     # Alternate placement of labels for overlapping points
#                     if j % 2 == 0:
#                         plt.text(x + 0.1, y, batch_label, fontsize=annotation_font_size, ha='left', va='center')
#                     else:
#                         plt.text(x - 0.1, y, batch_label, fontsize=annotation_font_size, ha='right', va='center')
#                 else:
#                     plt.text(x + 0.1, y, batch_label, fontsize=annotation_font_size, ha='left', va='center')
#
#         # Overlay centralized and federated data points
#         for i, dataset in enumerate(datasets):
#             # Centralized as horizontal lines only within the dataset range
#             if not scgpt[scgpt['Dataset'] == dataset].empty:
#                 scgpt_value = scgpt[scgpt['Dataset'] == dataset]['Value'].values[0]
#                 plt.hlines(y=scgpt_value, xmin=i + 0.7, xmax=i + 1.3,
#                            color=colors[i % len(colors)], linestyle='--', linewidth=2, zorder=3,
#                            label=f"scGPT")
#
#             # Federated as scatter points
#             if not fedscgpt[fedscgpt['Dataset'] == dataset].empty:
#                 federated_value = fedscgpt[fedscgpt['Dataset'] == dataset]['Value'].values[0]
#                 plt.scatter(i + 1, federated_value, color=colors[i % len(colors)], edgecolor='black',
#                             zorder=5, marker=FEDSCGPT_MARKER, s=100, label=f"FedscGPT")
#
#             if not fedscgpt_smpc[fedscgpt_smpc['Dataset'] == dataset].empty:
#                 federated_value = fedscgpt_smpc[fedscgpt_smpc['Dataset'] == dataset]['Value'].values[0]
#                 plt.scatter(i + 1, federated_value, color=colors[i % len(colors)], edgecolor='black',
#                             zorder=5, marker=FEDSCGPT_SMPC_MARKER, s=100, label=f"FedscGPT-SMPC")
#
#
#
#         # Customize the plot
#         plt.xlabel('', fontsize=1)
#         plt.ylabel(metric.capitalize(), fontsize=20)
#         plt.ylim(0, 1)
#         plt.xticks(range(1, len(datasets) + 1), [handle_ds_name(d) for d in datasets], fontsize=20)
#         plt.yticks(fontsize=18)
#
#         # Legend placement based on the flag
#         custom_handles = [
#             plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='white', markersize=8, linestyle='None',
#                         label='Clients'),
#             plt.Line2D([0], [0], marker=FEDSCGPT_MARKER, color='black', markerfacecolor='white', markersize=10, linestyle='None',
#                         label='FedscGPT'),
#             plt.Line2D([0], [0], marker=FEDSCGPT_SMPC_MARKER, color='black', markerfacecolor='white', markersize=10, linestyle='None',
#                        label='FedscGPT-SMPC'),
#             plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='scGPT')
#         ]
#         legend = plt.legend(handles=custom_handles, loc='lower left', fontsize=14, frameon=True)
#         legend.get_frame().set_edgecolor('black')
#         legend.get_frame().set_facecolor('white')
#
#         plt.tight_layout()
#         plt.savefig(f"{plots_dir}/{metric}_scatterplot_annotated.{img_format}", format=img_format, dpi=300,
#                     bbox_inches='tight')
#         plt.close()

def print_best_configurations(df, metrics):
    for metric in metrics:
        print(f"\n=== Best configurations for Metric: {metric} ===")
        sub_df = df[df['Metric'] == metric]

        grouped = sub_df.groupby(['Aggregation', 'Dataset'])

        for (agg, ds), group in grouped:
            best_row = group.loc[group['Value'].idxmax()]
            print(
                f"- {agg} | {ds} -> Value: {best_row['Value']:.4f} | Epochs: {int(best_row['n_epochs'])} | Round: {int(best_row['Round'])}")


import matplotlib.pyplot as plt


def save_legend_figure(handles, labels, save_path="legend_only.svg", fontsize=14, ncol=1):
    """
    Save only the legend as an independent figure in one row.

    Parameters:
    - handles: The plot handles (such as from plot, scatter, etc.)
    - labels: The corresponding labels for each handle
    - save_path: The path to save the legend figure (default is "legend_only.svg")
    - fontsize: Font size for the labels in the legend (default is 14)
    - ncol: Number of columns in the legend (default is 1)
    """
    # Adjust the width of the figure based on the number of columns
    ncol = max(1, ncol)  # Ensure ncol is at least 1
    fig_width = max(1.5 * len(handles) / ncol, 6)  # Scale width based on ncol
    fig_legend = plt.figure(figsize=(fig_width, 2))  # Wider but short figure for the legend

    # Create the legend outside of any axes
    legend = fig_legend.legend(
        handles,
        labels,
        loc='center',
        frameon=True,
        fontsize=fontsize,
        ncol=ncol,  # Use the passed ncol value
        handletextpad=0.1,
        columnspacing=0.3,
        borderaxespad=0.1
    )

    # Customize legend frame
    legend.get_frame().set_edgecolor('none')
    legend.get_frame().set_facecolor('white')

    # Tight layout and save the figure
    fig_legend.tight_layout()
    fig_legend.savefig(save_path, format="png", dpi=300, bbox_inches='tight')
    plt.close(fig_legend)


# Example usage:
# save_legend_figure(handles, labels, ncol=2)  # To save the legend with 2 columns


def accuracy_annotated_scatterplot(df, plots_dir, img_format='svg', proximity_threshold=0.1):
    # Constants
    valid_datasets = ['hp5', 'ms', 'lung', 'cl', 'myeloid-top5']
    fed_aggs = ['weighted-FedAvg', 'SMPC-weighted-FedAvg', 'weighted-FedProx', 'SMPC-weighted-FedProx']
    client_aggs = [agg for agg in df['Aggregation'].unique() if agg.startswith("client_")]
    all_aggs = ['centralized'] + fed_aggs + client_aggs

    # Filtering
    df = df[df['Dataset'].isin(valid_datasets) & df['Aggregation'].isin(all_aggs)]

    # df = df.groupby(['Dataset', 'Metric', 'Aggregation'], as_index=False).apply(
    #     lambda g: g.loc[g['Value'].idxmax()]).reset_index(drop=True)

    fixed_df = df[(df['n_epochs'] == 1) & (df['Round'] == 20)].copy()

    # Step 2: Group and keep one per Dataset-Metric-Aggregation
    fixed_df = fixed_df.groupby(['Dataset', 'Metric', 'Aggregation'], as_index=False).first()

    # Step 3: Add additional client and centralized rows (if not already in fixed_df)
    # Select all client_* and centralized entries
    extra_df = df[df['Aggregation'].str.startswith('client_') | (df['Aggregation'] == 'centralized')]


    # Remove duplicates already in fixed_df
    existing_keys = set(fixed_df[['Dataset', 'Metric', 'Aggregation']].apply(tuple, axis=1))
    extra_df = extra_df[~extra_df[['Dataset', 'Metric', 'Aggregation']].apply(tuple, axis=1).isin(existing_keys)]

    # Step 4: Concatenate
    df = pd.concat([fixed_df, extra_df], ignore_index=True)



    # Output dir
    os.makedirs(plots_dir, exist_ok=True)

    metrics = df['Metric'].unique()
    # ds_map = {'hp5': 'hp', 'myeloid-top5': 'myeloid'}
    # df['Dataset'] = df['Dataset'].apply(lambda x: ds_map.get(x, x))
    datasets = sorted(df['Dataset'].unique())
    palette = generate_palette(datasets)

    agg_map = {
        "weighted-FedAvg": ('^', "FedAvg", -0.2),
        "SMPC-weighted-FedAvg": ('^', "FedAvg-SMPC", -0.1),
        "weighted-FedProx": ('s', "FedProx", 0.1),
        "SMPC-weighted-FedProx": ('s', "FedProx-SMPC", 0.2),
    }

    for metric in metrics:
        plt.figure(figsize=(6, 4))
        df_metric = df[df['Metric'] == metric]

        for i, dataset in enumerate(valid_datasets):
            subset = df_metric[df_metric['Dataset'] == dataset]
            color = palette[dataset]
            # Plot client data
            clients = subset[subset['Aggregation'].str.startswith("client_")]
            jitter = np.random.uniform(-0.1, 0.1, size=len(clients))
            x_vals = i + 1 + jitter
            y_vals = clients['Value'].values
            labels = clients['Aggregation'].str.replace("client_", "", regex=False).values

            plt.scatter(x_vals, y_vals, color=color, edgecolor='black', s=50, alpha=0.7,
                        label=f"Client Data - {dataset}" if i == 0 else None)

            # for j, (x, y, lab) in enumerate(zip(x_vals, y_vals, labels)):
            #     offset = 0.1 if j % 2 == 0 else -0.1
            #     ha = 'left' if offset > 0 else 'right'
            #     plt.text(x + offset, y, shorten_batch_value(lab), fontsize=11, ha=ha, va='center')

            # Centralized
            centralized = subset[subset['Aggregation'] == 'centralized']
            if not centralized.empty:
                val = centralized['Value'].values[0]
                plt.hlines(val, i + 0.7, i + 1.3, color=color, linestyle='--', linewidth=2,
                           label="Centralized" if i == 0 else None)

            for agg, (marker, label_text, offset) in agg_map.items():
                row = subset[subset['Aggregation'] == agg]
                if not row.empty:
                    val = row['Value'].values[0]
                    epoch = int(row['n_epochs'].values[0])
                    rnd = int(row['Round'].values[0])
                    x_pos = i + 1 + offset

                    # Use unfilled face for SMPC variants
                    fill_style = 'white' if 'SMPC' not in agg else color
                    edge = 'black'

                    plt.scatter(x_pos, val, marker=marker, s=100,
                                facecolors=fill_style, edgecolors=edge,
                                zorder=5, label=label_text if i == 0 else None)

                    # Annotate (epoch, round)
                    # plt.text(x_pos, val + 0.02, f"{epoch}e\n{rnd}r", fontsize=9, ha='center', va='bottom')

        # Styling
        plt.ylim(0.0, 1.1)
        plt.ylabel(metric.capitalize(), fontsize=16)
        plt.xticks(ticks=range(1, len(valid_datasets) + 1),
                   labels=[handle_ds_name(d) for d in valid_datasets],
                   fontsize=20)
        plt.yticks(fontsize=13)

        # Save figure
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{metric}_scatterplot.{img_format}", format=img_format, dpi=300)
        plt.close()

        # Save external legend
        custom_handles = [
            plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='white',
                       markersize=8, linestyle='None', label='Clients'),
            plt.Line2D([0], [0], marker='^', color='black', markerfacecolor='white', markersize=9,
                       linestyle='None', label='FedAvg'),
            plt.Line2D([0], [0], marker='^', color='black', markerfacecolor='black', markersize=9,
                       linestyle='None', label='FedAvg-SMPC'),
            plt.Line2D([0], [0], marker='s', color='black', markerfacecolor='white', markersize=9,
                       linestyle='None', label='FedProx'),
            plt.Line2D([0], [0], marker='s', color='black', markerfacecolor='black', markersize=9,
                       linestyle='None', label='FedProx-SMPC'),
            plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Centralized'),
        ]

        save_legend_figure(custom_handles, [h.get_label() for h in custom_handles],
                           save_path=f"{plots_dir}/legend_only.{img_format}", fontsize=20)


def myeloid_scatterplot(df, plots_dir, img_format='png'):
    sns.set(style='whitegrid')
    os.makedirs(plots_dir, exist_ok=True)

    valid_datasets = ['Top5', 'Top10', 'Top20', 'Top30']
    target_agg = "SMPC-weighted-FedProx"
    client_aggs = [agg for agg in df['Aggregation'].unique() if agg.startswith("client_")]
    all_aggs = ['centralized', target_agg] + client_aggs

    # Filter relevant rows
    df = df[df['Dataset'].isin(valid_datasets) & df['Aggregation'].isin(all_aggs)]

    palette = generate_palette(valid_datasets)
    agg_map = {
        target_agg: ('s', "FedProx-SMPC", 0.),
    }

    metrics = df['Metric'].unique()

    # Find best rounds for each dataset (using Macro_F1)
    best_rounds = {}
    for dataset in valid_datasets:
        best_macro = df[
            (df['Dataset'] == dataset) &
            (df['Aggregation'] == target_agg) &
            (df['Metric'] == 'Macro-F1') &
            (df['n_epochs'] == 1) &
            (df['mu'] == 0.01)
        ]
        if not best_macro.empty:
            best_row = best_macro.loc[best_macro['Value'].idxmax()]
            best_rounds[dataset] = int(best_row['Round'])
        else:
            raise ValueError
    print(best_rounds)
    for metric in metrics:
        plt.figure(figsize=(5, 4))
        df_metric = df[df['Metric'] == metric]

        for i, dataset in enumerate(valid_datasets):
            subset = df_metric[df_metric['Dataset'] == dataset]
            color = palette[dataset]

            # --- Client points ---
            clients = subset[subset['Aggregation'].str.startswith("client_")]
            jitter = np.random.uniform(-0.1, 0.1, size=len(clients))
            x_vals = i + 1 + jitter
            y_vals = clients['Value'].values
            plt.scatter(x_vals, y_vals, color=color, edgecolor='black', s=50, alpha=0.7)

            # --- FedProx-SMPC (only best round per Macro_F1) ---
            for agg, (marker, label_text, offset) in agg_map.items():
                best_rnd = best_rounds.get(dataset)
                if best_rnd is not None:
                    fed_row = subset[
                        (subset['Aggregation'] == agg) &
                        (subset['Round'] == best_rnd) &
                        (subset['n_epochs'] == 1) &
                        (subset['mu'] == 0.01)
                    ]
                    if not fed_row.empty:
                        val = fed_row['Value'].values[0]
                        x_pos = i + 1 + offset
                        face_color = color

                        plt.scatter(x_pos, val, marker=marker, s=100,
                                    facecolors=face_color, edgecolors='black', zorder=5)

        # --- Centralized line for Top30 ---
        centralized = df_metric[(df_metric['Dataset'] == 'Top30') & (df_metric['Aggregation'] == 'centralized')]
        if not centralized.empty:
            val = centralized['Value'].values[0]
            plt.axhline(val, color='black', linestyle='--', linewidth=2)

        # --- Axes ---
        plt.ylim(0.0, .7)
        plt.ylabel(metric, fontsize=16)
        plt.xticks(
            ticks=range(1, len(valid_datasets) + 1),
            labels=[handle_ds_name(d) for d in valid_datasets],
            fontsize=16
        )
        plt.yticks(fontsize=13)

        # --- Save ---
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{metric}_scatterplot.{img_format}", format=img_format, dpi=300)
        plt.close()
        custom_handles = [
            plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='white',
                       markersize=8, linestyle='None', label='Clients'),
            plt.Line2D([0], [0], marker='s', color='black', markerfacecolor='black', markersize=9,
                       linestyle='None', label='FedProx-SMPC'),
            plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Centralized'),
        ]

        save_legend_figure(custom_handles, [h.get_label() for h in custom_handles],
                           save_path=f"{plots_dir}/legend_3_only.{img_format}", fontsize=20,
                           ncol=3)


def covid_scatterplot(df, plots_dir, img_format='png'):
    os.makedirs(plots_dir, exist_ok=True)

    fed_aggs = ['FedAvg', 'FedAvg-SMPC', 'FedProx', 'FedProx-SMPC']
    client_aggs = [agg for agg in df['Aggregation'].unique() if agg.startswith("client_")]
    aggregations = ['centralized'] + fed_aggs + client_aggs

    datasets = df['Dataset'].unique()
    metrics = df['Metric'].unique()
    palette = generate_palette(datasets)

    agg_map = {
        "FedAvg": ('^', "FedAvg", -0.2),
        "FedAvg-SMPC": ('^', "FedAvg-SMPC", -0.1),
        "FedProx": ('s', "FedProx", 0.1),
        "FedProx-SMPC": ('s', "FedProx-SMPC", 0.2),
    }

    fig, axes = plt.subplots(len(metrics), 1, figsize=(4 , 3 * len(metrics)), sharex=True)

    if len(metrics) == 1:
        axes = [axes]  # Ensure iterable

    for ax_i, metric in zip(axes, metrics):
        df_metric = df[df['Metric'] == metric]

        for i, dataset in enumerate(datasets):
            subset = df_metric[df_metric['Dataset'] == dataset]
            color = palette[dataset]

            # Clients
            clients = subset[subset['Aggregation'].str.startswith("client_")]
            jitter = np.random.uniform(-0.1, 0.1, size=len(clients))
            x_vals = i + 1 + jitter
            y_vals = clients['Value'].values
            ax_i.scatter(x_vals, y_vals, color=color, edgecolor='black', s=50, alpha=0.7)

            # Centralized
            centralized = subset[subset['Aggregation'] == 'centralized']
            if not centralized.empty:
                val = centralized['Value'].values[0]
                ax_i.hlines(val, i + 0.7, i + 1.3, color=color, linestyle='--', linewidth=2)
            print(f"Dataset: {dataset}, Metric: {metric}, Centralized Value: {val:.2f}")
            # Federated methods
            for agg, (marker, label_text, offset) in agg_map.items():
                row = subset[(subset['Aggregation'] == agg) & (subset['n_epochs'] == 1) & (subset['Round'] == 20)]
                assert len(row) == 1, f"Expected one row for {agg} in {dataset}, got {len(row)}: \n{row}"
                val = row['Value'].values[0]
                print(f"Dataset: {dataset}, Aggregation: {agg}, Metric: {metric}, Value: {val:.2f}")
                x_pos = i + 1 + offset
                fill_style = 'white' if 'SMPC' not in agg else color
                edge = 'black'

                ax_i.scatter(x_pos, val, marker=marker, s=100,
                             facecolors=fill_style, edgecolors=edge, zorder=5)

        ax_i.set_ylim(0.01, 1 if metric == 'Accuracy' else 0.8)
        ax_i.set_ylabel(metric.replace('_', '-'), fontsize=18)
        ax_i.tick_params(axis='y', labelsize=14)

    # Final adjustments
    axes[-1].set_xticks(range(1, len(datasets) + 1))
    axes[-1].set_xticklabels([d for d in datasets], fontsize=18)
    axes[-1].tick_params(axis='x', labelsize=18)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.02)
    fig.savefig(f"{plots_dir}/all_metrics_combined.{img_format}", dpi=300, format=img_format)
    plt.close()

def plot_best_per_metric_covid(df, subdir='efficiency_covid_v2'):
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    sns.set(style='whitegrid')
    os.makedirs(f"{ANNOTATION_PLOTS_DIR}/{subdir}", exist_ok=True)

    aggregations = ['centralized', 'FedAvg', 'FedAvg-SMPC', 'FedProx', 'FedProx-SMPC']
    metrics = ['Accuracy', 'Precision', 'Recall', 'Macro_F1']
    datasets = sorted(df['Dataset'].unique())

    palette = generate_palette(aggregations)
    palette['centralized'] = 'black'

    for metric in metrics:
        print(f"\n=== Metric: {metric} ===")
        plot_data = []
        legend_labels = {}

        for dataset in datasets:
            for agg in aggregations:
                sub_df = df[
                    (df['Dataset'] == dataset) &
                    (df['Aggregation'] == agg) &
                    (df['Metric'] == metric)
                ]

                if sub_df.empty:
                    print(f"  Skipping {dataset} - {agg} (no data)")
                    continue

                best_idx = sub_df['Value'].idxmax()
                best_row = df.loc[best_idx]
                best_round = best_row['Round']
                best_epoch = best_row['n_epochs']
                best_mu = best_row['mu']

                print(f"  {dataset[:25]:25s} | {agg:15s} → Round: {best_round}, Epochs: {int(best_epoch)}, Mu: {best_mu}, Value: {best_row['Value']:.4f}")

                legend_labels[agg] = f"{agg} (round {best_round})"

                plot_data.append({
                    'Dataset': dataset,
                    'Aggregation': agg,
                    'Value': best_row['Value']
                })

        plot_df = pd.DataFrame(plot_data)
        plot_df['Aggregation'] = pd.Categorical(plot_df['Aggregation'], categories=aggregations, ordered=True)

        # Barplot for this metric grouped by dataset
        plt.figure(figsize=(max(8, len(datasets) * 1.2), 5))
        ax = sns.barplot(data=plot_df, x='Dataset', y='Value', hue='Aggregation', palette=palette)
        ax.set_title(metric, fontsize=20)
        ax.set_ylabel("Score", fontsize=18)
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', labelsize=12, rotation=25)
        ax.tick_params(axis='y', labelsize=14)
        plt.legend().remove()
        plt.tight_layout()
        plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{subdir}/bar_grouped_by_dataset_{metric.lower()}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Legend
        handles = [Patch(facecolor=palette[m], label=legend_labels.get(m, m)) for m in aggregations if m in plot_df['Aggregation'].unique()]
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.axis('off')
        ax.legend(handles=handles, loc='center', ncol=len(handles), frameon=False,
                  fontsize=20, handletextpad=0.4, columnspacing=0.8)
        plt.tight_layout()
        plt.savefig(f"{ANNOTATION_PLOTS_DIR}/{subdir}/legend_horizontal_{metric.lower()}.png", dpi=300, bbox_inches='tight', transparent=True)
        plt.close()

def plot_combined_metrics_per_dataset(df, subdir='efficiency_covid_combined'):
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    sns.set(style='whitegrid')
    os.makedirs(f"{ANNOTATION_PLOTS_DIR}/{subdir}", exist_ok=True)

    aggregations = ['centralized', 'FedAvg', 'FedAvg-SMPC', 'FedProx', 'FedProx-SMPC']
    metrics = ['Accuracy', 'Precision', 'Recall', 'Macro_F1']
    datasets = sorted(df['Dataset'].unique())

    palette = generate_palette(aggregations)
    palette['centralized'] = 'black'

    fig, axs = plt.subplots(1, len(metrics), figsize=(len(metrics) * 5.5, 5), sharey=True)

    for i, metric in enumerate(metrics):
        print(f"\n=== Metric: {metric} ===")
        plot_data = []
        legend_labels = {}

        for dataset in datasets:
            for agg in aggregations:
                sub_df = df[
                    (df['Dataset'] == dataset) &
                    (df['Aggregation'] == agg) &
                    (df['Metric'] == metric)
                ]

                if sub_df.empty:
                    print(f"  Skipping {dataset} - {agg} (no data)")
                    continue

                best_idx = sub_df['Value'].idxmax()
                best_row = df.loc[best_idx]
                best_round = best_row['Round']
                best_epoch = best_row['n_epochs']
                best_mu = best_row['mu']

                print(f"  {dataset[:25]:25s} | {agg:15s} → Round: {best_round}, Epochs: {int(best_epoch)}, Mu: {best_mu}, Value: {best_row['Value']:.4f}")

                legend_labels[agg] = f"{agg} (round {best_round})"

                plot_data.append({
                    'Dataset': dataset,
                    'Aggregation': agg,
                    'Value': best_row['Value']
                })

        plot_df = pd.DataFrame(plot_data)
        plot_df['Aggregation'] = pd.Categorical(plot_df['Aggregation'], categories=aggregations, ordered=True)

        ax = axs[i]
        sns.barplot(data=plot_df, x='Dataset', y='Value', hue='Aggregation', palette=palette, ax=ax)
        ax.set_title(metric, fontsize=20)
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("Score", fontsize=18)
        else:
            ax.set_ylabel("")
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', labelsize=12, rotation=30)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend_.remove()

    plt.tight_layout()
    fig.savefig(f"{ANNOTATION_PLOTS_DIR}/{subdir}/combined_bar_metrics_by_dataset.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save shared legend
    handles = [Patch(facecolor=palette[m], label=m) for m in aggregations]
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.axis('off')
    ax.legend(handles=handles, loc='center', ncol=len(handles), frameon=False,
              fontsize=20, handletextpad=0.4, columnspacing=0.8)
    plt.tight_layout()
    fig.savefig(f"{ANNOTATION_PLOTS_DIR}/{subdir}/combined_legend_horizontal.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.close()




def plot_batch_effect_umaps(raw_h5ad, cent_corrected, fed_corrected, batch_key, cell_key, out_prefix, standalone=False):
    sc.settings.figdir = ""
    # Load AnnData objects
    names = ["Raw", "Centralized", "Federated"]
    paths = [raw_h5ad, cent_corrected, fed_corrected]
    adatas = {}
    for name, path in zip(names, paths):
        adata = sc.read_h5ad(path)
        print(adata.obsm.keys())
        if "X_umap" not in adata.obsm:
            sc.pp.neighbors(adata, use_rep="X", n_neighbors=30)
            sc.tl.umap(adata)
            adata.write(path)
        adatas[name] = adata

    uniq_ct = list(adatas["Centralized"].obs[cell_key].cat.categories)
    cell_palette = generate_palette(uniq_ct)

    uniq_batch = list(adatas["Raw"].obs[batch_key].cat.categories)
    batch_palette = generate_palette(uniq_batch)

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    # Save separate UMAPs without legends
    for name in names:
        ad = adatas[name]

        # Cell type plot
        sc.pl.umap(ad, color=cell_key, show=False, legend_loc=None,
                   palette=cell_palette, title=f"{name} (cell type)")
        plt.savefig(f"{out_prefix}_{name}_celltype.png", dpi=300)
        plt.close()

        # Batch plot
        sc.pl.umap(ad, color=batch_key, show=False, legend_loc=None,
                   palette=batch_palette, title=f"{name} (batch)")
        plt.savefig(f"{out_prefix}_{name}_batch.png", dpi=300)
        plt.close()

    # Create separate legend for cell type
    ct_handles = [
        plt.Line2D([0], [0], marker='o', color=cell_palette[ct], linestyle='', label=shorten_celltype_value(ct))
        for ct in uniq_ct
    ]
    fig_ct, ax_ct = plt.subplots(figsize=(4, max(4, len(ct_handles) * 0.25)))
    ax_ct.legend(handles=ct_handles, loc='center', title=cell_key, fontsize='small', title_fontsize='medium')
    ax_ct.axis("off")
    plt.tight_layout()
    fig_ct.savefig(f"{out_prefix}_legend_celltype.png", dpi=300)
    plt.close(fig_ct)

    # Create separate legend for batch
    bt_handles = [
        plt.Line2D([0], [0], marker='o', color=batch_palette[b], linestyle='', label=shorten_batch_value(b))
        for b in uniq_batch
    ]
    fig_bt, ax_bt = plt.subplots(figsize=(4, max(4, len(bt_handles) * 0.25)))
    ax_bt.legend(handles=bt_handles, loc='center', title=batch_key, fontsize='small', title_fontsize='medium')
    ax_bt.axis("off")
    plt.tight_layout()
    fig_bt.savefig(f"{out_prefix}_legend_batch.png", dpi=300)
    plt.close(fig_bt)

def investigate_general(df):
    # Filter utility functions
    def get_fedavg(df, smpc):
        method = "SMPC-weighted-FedAvg" if smpc else "weighted-FedAvg"
        return df[df['Aggregation'] == method]

    def get_fedprox(df, smpc):
        method = "SMPC-weighted-FedProx" if smpc else "weighted-FedProx"
        return df[df['Aggregation'] == method]

    # Initialize Excel writer
    output_path = "federated_vs_central_analysis.xlsx"
    writer = pd.ExcelWriter(f"{ANNOTATION_PLOTS_DIR}/summary/{output_path}", engine='xlsxwriter')

    # -----------------------------------------------
    # Question 1: Federated without SMPC vs Federated with SMPC
    results_q1 = []
    datasets = df['Dataset'].unique()
    metrics = df['Metric'].unique()

    for dataset in datasets:
        for metric in metrics:
            smpc_df = get_fedavg(df, smpc=True)
            nosmpc_df = get_fedavg(df, smpc=False)

            smpc_score = smpc_df[(smpc_df['Dataset'] == dataset) & (smpc_df['Metric'] == metric)].sort_values("Value", ascending=False)
            nosmpc_score = nosmpc_df[(nosmpc_df['Dataset'] == dataset) & (nosmpc_df['Metric'] == metric)].sort_values("Value", ascending=False)

            if not smpc_score.empty and not nosmpc_score.empty:
                best_smpc = smpc_score.iloc[0]['Value']
                best_nosmpc = nosmpc_score.iloc[0]['Value']
                if best_nosmpc > best_smpc:
                    results_q1.append([dataset, metric, best_nosmpc, best_smpc])

    df_q1 = pd.DataFrame(results_q1, columns=['Dataset', 'Metric', 'No SMPC FedAvg', 'SMPC FedAvg'])
    print("Q1: Datasets where Federated without SMPC outperforms with SMPC:")
    print(df_q1.round(2))
    df_q1.round(2).to_excel(writer, sheet_name='Q1_NoSMPC_vs_SMPC', index=False)

    # -----------------------------------------------
    # Question 2: FedProx (SMPC) vs FedAvg (SMPC)
    results_q2 = []

    for dataset in datasets:
        for metric in metrics:
            fedprox = get_fedprox(df, smpc=True)
            fedavg = get_fedavg(df, smpc=True)

            fprox = fedprox[(fedprox['Dataset'] == dataset) & (fedprox['Metric'] == metric)].sort_values("Value", ascending=False)
            favg = fedavg[(fedavg['Dataset'] == dataset) & (fedavg['Metric'] == metric)].sort_values("Value", ascending=False)

            if not fprox.empty and not favg.empty:
                if fprox.iloc[0]['Value'] > favg.iloc[0]['Value']:
                    results_q2.append([dataset, metric, fprox.iloc[0]['Value'], favg.iloc[0]['Value']])

    df_q2 = pd.DataFrame(results_q2, columns=['Dataset', 'Metric', 'FedProx SMPC', 'FedAvg SMPC'])
    print("\nQ2: Datasets where FedProx (SMPC) > FedAvg (SMPC):")
    print(df_q2.round(2))
    df_q2.round(2).to_excel(writer, sheet_name='Q2_Prox_vs_Avg_SMP', index=False)

    # -----------------------------------------------
    # Question 3: FedProx (No SMPC) vs FedAvg (No SMPC)
    results_q3 = []

    for dataset in datasets:
        for metric in metrics:
            fedprox = get_fedprox(df, smpc=False)
            fedavg = get_fedavg(df, smpc=False)

            fprox = fedprox[(fedprox['Dataset'] == dataset) & (fedprox['Metric'] == metric)].sort_values("Value", ascending=False)
            favg = fedavg[(fedavg['Dataset'] == dataset) & (fedavg['Metric'] == metric)].sort_values("Value", ascending=False)

            if not fprox.empty and not favg.empty:
                if fprox.iloc[0]['Value'] > favg.iloc[0]['Value']:
                    results_q3.append([dataset, metric, fprox.iloc[0]['Value'], favg.iloc[0]['Value']])

    df_q3 = pd.DataFrame(results_q3, columns=['Dataset', 'Metric', 'FedProx', 'FedAvg'])
    print("\nQ3: Datasets where FedProx > FedAvg (no SMPC):")
    print(df_q3.round(2))
    df_q3.round(2).to_excel(writer, sheet_name='Q3_Prox_vs_Avg_NoSMP', index=False)

    # -----------------------------------------------
    # Question 4: Max federated shortcoming vs centralized
    results_q4 = []
    centralized = df[df['Aggregation'] == 'centralized']

    for dataset in datasets:
        for metric in metrics:
            central = centralized[(centralized['Dataset'] == dataset) & (centralized['Metric'] == metric)]
            fed = df[(df['Round'].notna()) & (df['Dataset'] == dataset) & (df['Metric'] == metric)]
            if not central.empty and not fed.empty:
                max_diff = central.iloc[0]['Value'] - fed['Value'].max()
                results_q4.append([dataset, metric, central.iloc[0]['Value'], fed['Value'].max(), max_diff])

    df_q4 = pd.DataFrame(results_q4, columns=['Dataset', 'Metric', 'Centralized', 'Best Federated', 'Shortcoming'])
    print("\nQ4: Max shortcoming of best federated vs centralized:")
    print(df_q4.sort_values('Shortcoming', ascending=False).round(2))
    df_q4.round(2).to_excel(writer, sheet_name='Q4_Fed_vs_Central', index=False)

    # -----------------------------------------------
    # Question 5: Max shortcoming of *any* federated method vs centralized
    results_q5 = []

    for dataset in datasets:
        for metric in metrics:
            central = centralized[(centralized['Dataset'] == dataset) & (centralized['Metric'] == metric)]
            any_fed = df[(df['Aggregation'].str.contains('Fed')) & (df['Dataset'] == dataset) & (df['Metric'] == metric)]
            if not central.empty and not any_fed.empty:
                max_diff = central.iloc[0]['Value'] - any_fed['Value'].max()
                results_q5.append([dataset, metric, central.iloc[0]['Value'], any_fed['Value'].max(), max_diff])

    df_q5 = pd.DataFrame(results_q5, columns=['Dataset', 'Metric', 'Centralized', 'Best FedAny', 'Shortcoming'])
    print("\nQ5: Max shortcoming of any federated method vs centralized:")
    print(df_q5.sort_values('Shortcoming', ascending=False).round(2))
    df_q5.round(2).to_excel(writer, sheet_name='Q5_FedAny_vs_Central', index=False)

    # -----------------------------------------------
    # Question 6: Any client reaches centralized performance
    results_q6 = []
    client_df = df[df['Aggregation'].str.startswith("client_")]

    for dataset in datasets:
        for metric in metrics:
            central = centralized[(centralized['Dataset'] == dataset) & (centralized['Metric'] == metric)]
            clients = client_df[(client_df['Dataset'] == dataset) & (client_df['Metric'] == metric)]

            if not central.empty and not clients.empty:
                best_client = clients.sort_values('Value', ascending=False).iloc[0]
                if best_client['Value'] >= central.iloc[0]['Value']:
                    results_q6.append([
                        dataset, metric, best_client['Aggregation'],
                        best_client['Value'], central.iloc[0]['Value']
                    ])

    df_q6 = pd.DataFrame(results_q6, columns=['Dataset', 'Metric', 'Client', 'Client Value', 'Centralized Value'])
    print("\nQ6: Clients that reach or outperform centralized:")
    print(df_q6.round(2))
    df_q6.round(2).to_excel(writer, sheet_name='Q6_Client_vs_Central', index=False)

    # Save all results
    writer.close()
    print(f"\n✅ Results saved to {output_path}")




def investigate_detailed(df):
    def get_fedavg(df, smpc):
        method = "SMPC-weighted-FedAvg" if smpc else "weighted-FedAvg"
        return df[df['Aggregation'] == method]

    def get_fedprox(df, smpc):
        method = "SMPC-weighted-FedProx" if smpc else "weighted-FedProx"
        return df[df['Aggregation'] == method]

    def safe_sheet_name(name):
        return name[:31]  # Excel limit

    output_path = "federated_vs_central_analysis_detailed.xlsx"
    writer = pd.ExcelWriter(f"{ANNOTATION_PLOTS_DIR}/summary/{output_path}", engine='xlsxwriter')

    datasets = df['Dataset'].unique()
    metrics = df['Metric'].unique()
    centralized = df[df['Aggregation'] == 'centralized']
    client_df = df[df['Aggregation'].str.startswith("client_")]

    for metric in metrics:
        # 1. FedAvg No SMPC > FedAvg SMPC
        results = []
        for dataset in datasets:
            smpc_df = get_fedavg(df, True)
            nosmpc_df = get_fedavg(df, False)
            smpc_score = smpc_df[(smpc_df['Dataset'] == dataset) & (smpc_df['Metric'] == metric)]
            nosmpc_score = nosmpc_df[(nosmpc_df['Dataset'] == dataset) & (nosmpc_df['Metric'] == metric)]
            if not smpc_score.empty and not nosmpc_score.empty:
                best_smpc = smpc_score['Value'].max()
                best_nosmpc = nosmpc_score['Value'].max()
                if best_nosmpc > best_smpc:
                    results.append([dataset, round(best_nosmpc, 2), round(best_smpc, 2)])
        df1 = pd.DataFrame(results, columns=['Dataset', 'No SMPC FedAvg', 'SMPC FedAvg'])
        print(f"\n🟢 Metric: {metric} | FedAvg (No SMPC) > FedAvg (SMPC):")
        print(df1)
        df1.to_excel(writer, sheet_name=safe_sheet_name(f'FedAvg_NoSmpc_gt_Smpc_{metric}'), index=False)

        # 2. FedProx SMPC > FedAvg SMPC
        results = []
        for dataset in datasets:
            prox = get_fedprox(df, True)
            avg = get_fedavg(df, True)
            fprox = prox[(prox['Dataset'] == dataset) & (prox['Metric'] == metric)]
            favg = avg[(avg['Dataset'] == dataset) & (avg['Metric'] == metric)]
            if not fprox.empty and not favg.empty:
                max_prox = fprox['Value'].max()
                max_avg = favg['Value'].max()
                if max_prox > max_avg:
                    results.append([dataset, round(max_prox, 2), round(max_avg, 2)])
        df2 = pd.DataFrame(results, columns=['Dataset', 'FedProx SMPC', 'FedAvg SMPC'])
        print(f"\n🟢 Metric: {metric} | FedProx (SMPC) > FedAvg (SMPC):")
        print(df2)
        df2.to_excel(writer, sheet_name=safe_sheet_name(f'FedProx_gt_FedAvg_SMPC_{metric}'), index=False)

        # 3. FedProx No SMPC > FedAvg No SMPC
        results = []
        for dataset in datasets:
            prox = get_fedprox(df, False)
            avg = get_fedavg(df, False)
            fprox = prox[(prox['Dataset'] == dataset) & (prox['Metric'] == metric)]
            favg = avg[(avg['Dataset'] == dataset) & (avg['Metric'] == metric)]
            if not fprox.empty and not favg.empty:
                max_prox = fprox['Value'].max()
                max_avg = favg['Value'].max()
                if max_prox > max_avg:
                    results.append([dataset, round(max_prox, 2), round(max_avg, 2)])
        df3 = pd.DataFrame(results, columns=['Dataset', 'FedProx', 'FedAvg'])
        print(f"\n🟢 Metric: {metric} | FedProx (No SMPC) > FedAvg (No SMPC):")
        print(df3)
        df3.to_excel(writer, sheet_name=safe_sheet_name(f'FedProx_gt_FedAvg_NoSMPC_{metric}'), index=False)

        # 4. Federated best vs Centralized
        results = []
        for dataset in datasets:
            central = centralized[(centralized['Dataset'] == dataset) & (centralized['Metric'] == metric)]
            fed = df[(df['Round'].notna()) & (df['Dataset'] == dataset) & (df['Metric'] == metric)]
            if not central.empty and not fed.empty:
                best_central = central.iloc[0]['Value']
                best_fed = fed['Value'].max()
                shortcoming = round(best_central - best_fed, 2)
                results.append([dataset, round(best_central, 2), round(best_fed, 2), shortcoming])
        df4 = pd.DataFrame(results, columns=['Dataset', 'Centralized', 'Best Federated', 'Shortcoming'])
        print(f"\n📉 Metric: {metric} | Shortcomings of Best Federated vs Centralized:")
        print(df4.sort_values('Shortcoming', ascending=False))
        df4.to_excel(writer, sheet_name=safe_sheet_name(f'Fed_vs_Central_{metric}'), index=False)

        # 5. Any Fed method vs Centralized
        results = []
        for dataset in datasets:
            central = centralized[(centralized['Dataset'] == dataset) & (centralized['Metric'] == metric)]
            any_fed = df[(df['Aggregation'].str.contains("Fed")) & (df['Dataset'] == dataset) & (df['Metric'] == metric)]
            if not central.empty and not any_fed.empty:
                best_central = central.iloc[0]['Value']
                best_any = any_fed['Value'].max()
                shortcoming = round(best_central - best_any, 2)
                results.append([dataset, round(best_central, 2), round(best_any, 2), shortcoming])
        df5 = pd.DataFrame(results, columns=['Dataset', 'Centralized', 'Best FedAny', 'Shortcoming'])
        print(f"\n📉 Metric: {metric} | Shortcomings of Any Fed method vs Centralized:")
        print(df5.sort_values('Shortcoming', ascending=False))
        df5.to_excel(writer, sheet_name=safe_sheet_name(f'AnyFed_vs_Central_{metric}'), index=False)

        # 6. Clients matching Centralized
        results = []
        for dataset in datasets:
            central = centralized[(centralized['Dataset'] == dataset) & (centralized['Metric'] == metric)]
            clients = client_df[(client_df['Dataset'] == dataset) & (client_df['Metric'] == metric)]
            if not central.empty and not clients.empty:
                best_client = clients.sort_values('Value', ascending=False).iloc[0]
                if best_client['Value'] >= central.iloc[0]['Value']:
                    results.append([
                        dataset,
                        best_client['Aggregation'],
                        round(best_client['Value'], 2),
                        round(central.iloc[0]['Value'], 2)
                    ])
        df6 = pd.DataFrame(results, columns=['Dataset', 'Client', 'Client Value', 'Centralized Value'])
        print(f"\n📈 Metric: {metric} | Clients matching or outperforming Centralized:")
        print(df6)
        df6.to_excel(writer, sheet_name=safe_sheet_name(f'Client_vs_Central_{metric}'), index=False)

    writer.close()
    print(f"\n✅ Excel report saved to: {output_path}")


def summarize_best_hyperparams_by_metric(df):
    output_path = "best_hyperparams_per_federated_setting.xlsx"
    writer = pd.ExcelWriter(f"{ANNOTATION_PLOTS_DIR}/summary/{output_path}", engine='xlsxwriter')

    metrics = ['Accuracy', 'Precision', 'Recall', 'Macro_F1']
    federated_aggs = df[df['Round'].notna()]['Aggregation'].unique()
    datasets = df['Dataset'].unique()

    for metric in metrics:
        rows = []
        for dataset in datasets:
            for agg in federated_aggs:
                subdf = df[
                    (df['Dataset'] == dataset) &
                    (df['Aggregation'] == agg) &
                    (df['Metric'] == metric)
                ]
                if not subdf.empty:
                    best_row = subdf.loc[subdf['Value'].idxmax()]
                    rows.append({
                        'Dataset': dataset,
                        'Aggregation': agg,
                        'Best Value': round(best_row['Value'], 4),
                        'n_epochs': int(best_row['n_epochs']) if not pd.isna(best_row['n_epochs']) else 'NA',
                        'mu': round(best_row['mu'], 4) if not pd.isna(best_row['mu']) else 'NA',
                        'Round': int(best_row['Round']) if not pd.isna(best_row['Round']) else 'NA'
                    })

        result_df = pd.DataFrame(rows)
        print(f"\n📊 Best hyperparameters for metric: {metric}")
        print(result_df)
        sheet_name = f"BestParams_{metric}"[:31]
        result_df.to_excel(writer, sheet_name=sheet_name, index=False)

    writer.close()
    print(f"\n✅ Saved hyperparameter summary to: {output_path}")


def communication_efficiency_table(df):
    output_path = "communication_efficiency_summary.xlsx"
    writer = pd.ExcelWriter(f"{ANNOTATION_PLOTS_DIR}/summary/{output_path}", engine='xlsxwriter')

    thresholds = [0.70, 0.80, 0.90, 0.95, 0.99]
    metrics = df['Metric'].unique()
    datasets = df['Dataset'].unique()

    for metric in metrics:
        results = []
        for dataset in datasets:
            central_df = df[(df['Aggregation'] == 'centralized') &
                            (df['Dataset'] == dataset) &
                            (df['Metric'] == metric)]
            if central_df.empty:
                continue
            central_value = central_df.iloc[0]['Value']

            fed_df = df[(df['Dataset'] == dataset) &
                        (df['Metric'] == metric) &
                        (df['Round'].notna())]

            if fed_df.empty:
                continue

            for agg in fed_df['Aggregation'].unique():
                sub_df = fed_df[fed_df['Aggregation'] == agg]
                for threshold in thresholds:
                    cutoff = central_value * threshold
                    reached = sub_df[sub_df['Value'] >= cutoff]
                    if not reached.empty:
                        min_round = int(reached.sort_values('Round').iloc[0]['Round'])
                        epoch = int(reached.sort_values('Round').iloc[0]['n_epochs'])
                        mu = reached.sort_values('Round').iloc[0]['mu']
                        results.append({
                            'Dataset': dataset,
                            'Metric': metric,
                            'Aggregation': agg,
                            'Threshold': f"{int(threshold * 100)}%",
                            'Min Round': min_round,
                            'Epoch': epoch,
                            'mu': mu,
                            'Centralized Value': round(central_value, 4),
                            'Achieved Value': round(reached.sort_values('Round').iloc[0]['Value'], 4)
                        })

        df_metric = pd.DataFrame(results)
        print(f"\n📊 Communication efficiency for metric: {metric}")
        print(df_metric)
        sheet_name = f"CommEff_{metric}"[:31]
        df_metric.to_excel(writer, sheet_name=sheet_name, index=False)

    writer.close()
    print(f"\n✅ Saved communication efficiency summary to: {output_path}")


def plot_epoch_round_per_mu(df, metric="Accuracy", save_path="fedprox_epoch_round_by_mu.png"):
    """
        For each metric, plot a column of vertically stacked heatmaps (one per mu).
        Each heatmap is Epoch x Round (15 rounds max). μ is shown on the left.
        Shared colorbar per metric.
        """
    # Pre-filter
    df = df[
        (df['Dataset'] == 'ms') &
        (df['Aggregation'] == 'SMPC-weighted-FedProx')
        ].dropna(subset=['mu', 'n_epochs', 'Value', 'Round'])

    df['mu'] = df['mu'].astype(float)
    df['n_epochs'] = df['n_epochs'].astype(int)
    df['Round'] = df['Round'].astype(int)
    df = df[(df['Round'] > 0) & (df['Round'] <= 15)]

    metrics = df['Metric'].unique()
    mu_values = sorted(df['mu'].unique())
    num_metrics = len(metrics)
    num_mu = len(mu_values)

    fig, axs = plt.subplots(num_mu, num_metrics, figsize=(5 * num_metrics, 2 * num_mu), sharex='col')
    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.1, wspace=0.02, hspace=0.05)

    cmap = sns.color_palette("Blues", as_cmap=True)

    for col_idx, metric in enumerate(metrics):
        sub = df[df['Metric'] == metric]
        min_v = sub['Value'].min()
        max_v = 0.90  # or: max_v = sub['Value'].max()
        norm = colors.Normalize(vmin=min_v, vmax=max_v)

        mappable = None
        for row_idx, mu in enumerate(mu_values):
            ax = axs[row_idx][col_idx] if num_metrics > 1 else axs[row_idx]
            mu_df = sub[sub['mu'] == mu]
            pivot = mu_df.pivot_table(index='n_epochs', columns='Round', values='Value', aggfunc='mean')

            sns.heatmap(
                pivot, ax=ax, cmap=cmap, cbar=False,
                vmin=min_v, vmax=max_v,
                linewidths=0.1, linecolor='gray',
                annot=False
            )

            if mappable is None:
                mappable = ax.collections[0]

            # Annotate best cell
            if not pivot.empty:
                max_val = pivot.max().max()
                max_pos = np.where(pivot.values == max_val)
                if max_pos[0].size > 0:
                    i, j = max_pos[0][0], max_pos[1][0]
                    ax.text(j + 0.5, i + 0.5, f"{max_val:.2f}".lstrip("0"),
                            color='black', ha='center', va='center',
                            fontsize=10)

            # μ label on the left
            if col_idx == 0:
                ax.text(-0.22, 0.15, f"μ = {mu}", transform=ax.transAxes,
                        fontsize=14, va='bottom', ha='left', rotation=90)
                ax.set_ylabel("Epochs", fontsize=12)

            # Metric title on top
            if row_idx == 0:
                ax.set_title(metric, fontsize=14)

            # Remove ticks as needed
            if row_idx != num_mu - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Round", fontsize=14)

            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index.tolist(), fontsize=9, rotation=0)
            if col_idx != 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])



            if row_idx == num_mu - 1:
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns.tolist(), fontsize=9, rotation=0)

    # Shared colorbar for all
    cbar_ax = fig.add_axes([0.925, 0.3, 0.01, 0.4])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)

    plt.savefig(save_path, dpi=300)
    plt.close()



def find_and_combine_emb_metrics(base_dir="."):
    DATASETS = {
    "cl", "covid", "covid-corrected", "covid-fed-corrected", "hp5",
    "lung", "ms", "myeloid-top10", "myeloid-top20", "myeloid-top30", "myeloid-top5"
}

    AGGREGATION_CATEGORIES = {"centralized", "federated"}
    rows = []
    for root, dirs, files in os.walk(base_dir):
        if "evaluation_metrics.csv" in files:
            full_path = Path(os.path.join(root, "evaluation_metrics.csv"))
            aggregation = full_path.parent.name
            relative_parts = os.path.relpath(root, base_dir).split(os.sep)
            dataset = next((part for part in relative_parts if part in DATASETS), None)
            if not dataset:
                continue  # skip if dataset not found
            try:
                df = pd.read_csv(full_path)
                df.insert(0, "Aggregation", aggregation)
                df.insert(0, "Dataset", dataset)
                df.insert(0, "Source", os.path.relpath(root, base_dir))
                rows.append(df)
            except Exception as e:
                print(f"Error reading {full_path}: {e}")
    if rows:
        return pd.concat(rows, ignore_index=True)
    else:
        print("No evaluation_metrics.csv files found.")
        return pd.DataFrame()

def merge_result_pickles(data1, data2):
    merged = {}
    for source in [data1, data2]:
        for ds_name, ds_dict in source.items():
            merged.setdefault(ds_name, {})
            if 'id_maps' in ds_dict:
                if 'id_maps' in merged[ds_name]:
                    assert merged[ds_name]['id_maps'] == ds_dict['id_maps'], f"ID maps mismatch in dataset {ds_name}"
                else:
                    merged[ds_name]['id_maps'] = ds_dict['id_maps']
            for agg_method, agg_dict in ds_dict.items():
                if agg_method == 'id_maps':
                    continue
                merged[ds_name].setdefault(agg_method, {})
                for epoch, round_dict in agg_dict.items():
                    merged[ds_name][agg_method].setdefault(epoch, {})
                    for round_number, mu_dict in round_dict.items():
                        merged[ds_name][agg_method][epoch].setdefault(round_number, {})
                        for mu, res in mu_dict.items():
                            merged[ds_name][agg_method][epoch][round_number][mu] = res
    return merged

def structural_summary_df(temp):
    rows = []
    for dataset, dataset_dict in temp.items():
        if not isinstance(dataset_dict, dict):
            continue
        for agg_name, agg_dict in dataset_dict.items():
            if agg_name == 'id_maps':
                continue
            epochs = set()
            rounds = set()
            mus = set()
            for epoch, round_dict in agg_dict.items():
                epochs.add(epoch)
                for rnd, mu_dict in round_dict.items():
                    rounds.add(rnd)
                    for mu in mu_dict:
                        mus.add(mu)
            rows.append({
                'Dataset': dataset,
                'Aggregation': agg_name,
                'n_epochs': len(epochs),
                'max_epoch': max(epochs) if epochs else None,
                'n_rounds': len(rounds),
                'max_round': max(rounds) if rounds else None,
                'n_mu': len([m for m in mus if m is not None]),
                'max_mu': max([m for m in mus if m is not None], default=None)
            })
    df=pd.DataFrame(rows)
    cent = df[df.Aggregation == 'centralized'].copy()
    df = df[~(df.Aggregation == 'centralized')]
    clients = df[df.Aggregation.str.startswith('client_')].copy()
    df = df[~df.Aggregation.str.startswith('client_')]
    return cent, clients, df


