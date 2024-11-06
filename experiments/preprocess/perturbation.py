import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import scanpy as sc
from gears import PertData
import argparse
import os
import numpy as np
import anndata
import random

random.seed(42)
np.random.seed(42)


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default="/home/bba1658/FedscGPT/data/benchmark/perturbation")
parser.add_argument('--dataset', type=str, default="adamson", choices=["adamson", "norman"])
parser.add_argument('--n-clients', type=int, default=2)
parser.add_argument('--train-filename', type=str, default="perturb_processed.h5ad")
parser.add_argument('--test-filename', type=str, default="perturb_processed.h5ad")
parser.add_argument('--reverse', action='store_true', default=False)


def generate_condition_report(test_adata, train_val_adata, data_dir):
    """
    Generates a report logging the statistics about unique conditions in the test set vs train-validation set.
    Reports the number of unique one-gene perturbed and two-gene perturbed conditions in the test set.

    Parameters:
    - test_adata: AnnData object containing the test data.
    - train_val_adata: AnnData object containing the combined train and validation data.

    Returns:
    - A report dictionary containing the statistics.
    """
    report = {}

    # Extract unique conditions from test and train-validation datasets
    unique_test_conditions = test_adata.obs['condition'].unique()
    unique_train_val_conditions = train_val_adata.obs['condition'].unique()

    # Calculate the number of unique conditions in test and train-validation sets
    report['Number of unique conditions in test'] = len(unique_test_conditions)
    report['Number of unique conditions in train-validation'] = len(unique_train_val_conditions)

    # Identify one-gene perturbed (GeneA+ctrl) and two-gene perturbed (GeneA+GeneB) conditions in test set
    one_gene_perturbed_test = [cond for cond in unique_test_conditions if '+ctrl' in cond]
    two_genes_perturbed_test = [cond for cond in unique_test_conditions if '+' in cond and '+ctrl' not in cond]

    # Add these counts to the report
    report['Number of one-gene perturbed conditions in test'] = len(one_gene_perturbed_test)
    report['Number of two-genes perturbed conditions in test'] = len(two_genes_perturbed_test)

    # Log conditions that appear in test but not in train-validation and vice-versa
    test_only_conditions = set(unique_test_conditions) - set(unique_train_val_conditions)
    train_val_only_conditions = set(unique_train_val_conditions) - set(unique_test_conditions)

    report['Conditions unique to test'] = len(test_only_conditions)
    report['Conditions unique to train-validation'] = len(train_val_only_conditions)
    one_gene_unique_to_test = [cond for cond in test_only_conditions if '+ctrl' in cond]
    two_genes_unique_to_test = [cond for cond in test_only_conditions if '+' in cond and '+ctrl' not in cond]
    report['Number of one-gene perturbed conditions unique to test'] = len(one_gene_unique_to_test)
    report['Number of two-genes perturbed conditions unique to test'] = len(two_genes_unique_to_test)
    with open(f"{data_dir}/condition_report.log", 'w') as log_file:
        log_file.write("Condition Report\n")
        log_file.write("================\n")
        for key, value in report.items():
            log_file.write(f"{key}: {value}\n")
    # Create a detailed DataFrame for conditions
    condition_df = pd.DataFrame({
        'Condition': list(unique_test_conditions),
        'Perturbation Type': ['Two-Genes Perturbed' if '+' in cond and '+ctrl' not in cond else 'One-Gene Perturbed' for
                              cond in unique_test_conditions],
        'Set': ['Test' for _ in range(len(unique_test_conditions))]
    })

    # Save the condition DataFrame for detailed inspection
    condition_df.to_csv(f"{data_dir}/condition_report.csv", index=False)

    return report


def load_pert_data(data_dir, dataset):
    pert_data = PertData(data_dir)
    print(os.path.join(data_dir, dataset))
    pert_data.load(data_path=os.path.join(data_dir, dataset))
    return pert_data

def read_split(data_dir, dataset, filename):
    adata = is_data_preprocessed(data_dir, dataset)
    if not adata:
        pert_data = load_pert_data(data_dir, dataset)
        pert_data.prepare_split(seed=42)
        adata = pert_data.adata
        adata.uns['subgroups'] = pert_data.subgroup
        adata.write_h5ad(os.path.join(data_dir, dataset, filename))
    write_train_val_data(adata, data_dir, dataset, filename)
    train = adata[adata.obs.split == "train"].copy()
    val = adata[adata.obs.split == "val"].copy()
    return train, val

def write_train_val_data(adata, data_dir, dataset, filename):
    train_val_adata = adata[adata.obs.split != "test"].copy()
    centralized_train_dir = f"{data_dir}/{dataset}/centralized-train"
    save_split_data(train_val_adata, centralized_train_dir, filename)


def is_data_preprocessed(data_dir, dataset):
    adata = anndata.read_h5ad(os.path.join(data_dir, dataset, "perturb_processed.h5ad"))
    if "split" in adata.obs.columns and adata.uns:
        return adata
    return False

def split_homogeneous(adata, n_clients):
    unique_conditions = adata.obs.condition.unique()
    client_datasets = [adata[0:0].copy() for _ in range(n_clients)]  # Empty datasets for each client

    for condition in unique_conditions:
        condition_data = adata[adata.obs.condition == condition]
        condition_indices = np.arange(condition_data.shape[0])
        np.random.shuffle(condition_indices)

        # Split indices equally among clients
        split_indices = np.array_split(condition_indices, n_clients)

        for i, indices in enumerate(split_indices):
            client_datasets[i] = anndata.concat([client_datasets[i], condition_data[indices].copy()])
    return client_datasets


def split_heterogeneous(adata, n_clients):
    # Split 'ctrl' condition samples homogeneously across clients
    ctrl_data = adata[adata.obs.condition == "ctrl"]
    ctrl_indices = np.arange(ctrl_data.shape[0])
    np.random.shuffle(ctrl_indices)
    ctrl_splits = np.array_split(ctrl_indices, n_clients)

    # Prepare empty datasets for each client
    client_datasets = [adata[0:0].copy() for _ in range(n_clients)]  # Empty AnnData objects for each client

    # Assign the 'ctrl' condition samples to each client
    for i in range(n_clients):
        client_datasets[i] = ctrl_data[ctrl_splits[i]].copy()

    # Handle the remaining conditions (non-ctrl)
    other_data = adata[adata.obs.condition != "ctrl"]
    unique_conditions = other_data.obs.condition.unique()
    np.random.shuffle(unique_conditions)
    conditions_per_client = np.array_split(unique_conditions, n_clients)

    # Assign non-ctrl condition samples heterogeneously to each client
    for i, client_conditions in enumerate(conditions_per_client):
        client_datasets[i] = anndata.concat(
            [client_datasets[i], other_data[other_data.obs.condition.isin(client_conditions)].copy()])

    return client_datasets


def split_by_condition(train_adata, val_adata, n_clients, data_dir, filename):
    os.makedirs(data_dir, exist_ok=True)

    clients_validation = split_homogeneous(val_adata, n_clients)
    clients_train = split_heterogeneous(train_adata, n_clients)
    clients_data = []
    for i, (client_train, client_val) in enumerate(zip(clients_train, clients_validation)):
        client_dir = os.path.join(data_dir, f"{i + 1}")
        os.makedirs(client_dir, exist_ok=True)
        client_adata = anndata.concat([client_train, client_val])
        client_adata.uns = train_adata.uns.copy()
        client_adata.var = train_adata.var.copy()
        clients_data.append(client_adata)
        save_split_data(clients_data[-1], client_dir, filename)
        # clients_data[-1].write(os.path.join(client_dir, filename))
        print(f"Client {i + 1} - Train samples: {client_train.shape[0]}, Validation samples: {client_val.shape[0]} saved to {client_dir}")
    return clients_data


def plot_distribution(clients_data, data_dir):
    """
    Plots the distribution of samples and conditions across clients for both training and validation data.
    Colors y-tick labels green for conditions only in validation and red for conditions in more than one client.

    Parameters:
    - clients_data: List of AnnData objects, one for each client.
    - data_dir: Directory where the plot and CSV file will be saved.

    Returns:
    - None. Displays the plots and saves them to the specified directory.
    """
    if clients_data:
        data = []
        val_only_conditions = set()
        multi_client_conditions = set()

        # Dictionary to track the occurrence of conditions in clients
        condition_client_count = {}

        # Loop over each client dataset and aggregate the number of samples per condition
        for i, client_adata in enumerate(clients_data):
            condition_counts = client_adata.obs['condition'].value_counts().reset_index()
            condition_counts.columns = ['Condition', 'Number of Samples']
            condition_counts['Client'] = f'Client {i + 1}'
            data.append(condition_counts)

            # Track the conditions and their occurrences in different clients
            for condition in client_adata.obs['condition'].unique():
                if condition not in condition_client_count:
                    condition_client_count[condition] = 0
                condition_client_count[condition] += 1

                # Identify conditions only in validation
                if all(client_adata[client_adata.obs['condition'] == condition].obs['split'] == 'val'):
                    val_only_conditions.add(condition)

        # Identify conditions that appear in more than one client
        for condition, count in condition_client_count.items():
            if count > 1:
                multi_client_conditions.add(condition)

        # Concatenate all client data into a single DataFrame
        df = pd.concat(data, ignore_index=True)

        # Add columns to track condition status (whether it is validation-only or multi-client)
        df['Condition Status'] = df['Condition'].apply(lambda x: 'Validation Only' if x in val_only_conditions else (
            'Multi-Client' if x in multi_client_conditions else 'Single Client'))

        # Save the DataFrame to CSV
        df.to_csv(f"{data_dir}/unique_conditions_per_client.csv", index=False)
    else:
        df = pd.read_csv(f"{data_dir}/unique_conditions_per_client.csv")

    # Create the horizontal bar plot
    plt.figure(figsize=(10, 12))
    ax = sns.barplot(y='Condition', x='Number of Samples', hue='Client', data=df, orient='h')
    plt.title('Number of Samples per Condition Across Clients', fontsize=16)
    plt.ylabel('Condition', fontsize=14)
    plt.xlabel('Number of Samples', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Client', fontsize=12, title_fontsize=14)
    plt.tight_layout()

    # Change the color of y-axis tick labels based on the condition status
    for label in ax.get_yticklabels():
        condition = label.get_text()
        if condition in val_only_conditions:
            label.set_color('green')
        elif condition in multi_client_conditions:
            label.set_color('red')

    plt.savefig(f"{data_dir}/unique_conditions_per_client.png")


def load_clients_splits():
    clients_data = []
    for client in range(1, args.n_clients + 1):
        client_dir = os.path.join(args.data_dir, f"{args.dataset}/{args.n_clients}_clients/{client}")
        client_data = anndata.read_h5ad(os.path.join(client_dir, "train_val.h5ad"))
        clients_data.append(client_data)
    return clients_data

def save_split_data(adata, path, filename):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    assert adata.uns, "No UnStructured metadata found in the AnnData object."
    assert "gene_name" in adata.var.keys(), "No gene names found in the AnnData object variables."
    adata.write_h5ad(filepath)


def compare_pyg_objects(all_data_path, train_data_path, test_data_path):
    """
    Compare pyg objects created using all data, train data, and test data.

    Parameters:
    - all_data_path: Path to the pickle file for pyg object created with all data.
    - train_data_path: Path to the pickle file for pyg object created with train data.
    - test_data_path: Path to the pickle file for pyg object created with test data.

    Returns:
    - A dictionary summarizing the differences and similarities.
    """

    def load_pyg_object(file_path):
        pyg_object_name = os.path.join("data_pyg", "cell_graphs.pkl")
        file_path = os.path.join(file_path, pyg_object_name)
        """Load pyg object from a pickle file."""
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return None
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    # Load the pyg objects
    all_data_obj = load_pyg_object(all_data_path)
    train_data_obj = load_pyg_object(train_data_path)
    test_data_obj = load_pyg_object(test_data_path)

    # Initialize result dictionary to store comparison outcomes
    comparison_result = {}

    if all_data_obj and train_data_obj:
        comparison_result['All vs Train'] = compare_single_pyg_objects(all_data_obj, train_data_obj, 'All', 'Train')

    if all_data_obj and test_data_obj:
        comparison_result['All vs Test'] = compare_single_pyg_objects(all_data_obj, test_data_obj, 'All', 'Test')

    if train_data_obj and test_data_obj:
        comparison_result['Train vs Test'] = compare_single_pyg_objects(train_data_obj, test_data_obj, 'Train', 'Test')

    # Print the comparison report
    print_report(comparison_result)

    # Return the comparison summary
    return comparison_result


def compare_single_pyg_objects(obj1, obj2, obj1_name, obj2_name):
    """
    Compare two pyg objects and return differences and similarities.
    """
    comparison = {}

    # Compare the number of nodes
    comparison['num_nodes'] = {
        f'{obj1_name}': obj1.x.shape[0] if hasattr(obj1, 'x') else None,
        f'{obj2_name}': obj2.x.shape[0] if hasattr(obj2, 'x') else None,
        'match': (obj1.x.shape[0] == obj2.x.shape[0]) if hasattr(obj1, 'x') and hasattr(obj2, 'x') else False
    }

    # Compare the number of edges
    comparison['num_edges'] = {
        f'{obj1_name}': obj1.edge_index.shape[1] if hasattr(obj1, 'edge_index') else None,
        f'{obj2_name}': obj2.edge_index.shape[1] if hasattr(obj2, 'edge_index') else None,
        'match': (obj1.edge_index.shape[1] == obj2.edge_index.shape[1]) if hasattr(obj1, 'edge_index') and hasattr(obj2,
                                                                                                                   'edge_index') else False
    }

    # Compare node features (x)
    comparison['node_features_match'] = {
        'match': (obj1.x == obj2.x).all() if hasattr(obj1, 'x') and hasattr(obj2, 'x') else False
    }

    # Compare labels (y) if available
    comparison['labels_match'] = {
        'match': (obj1.y == obj2.y).all() if hasattr(obj1, 'y') and hasattr(obj2, 'y') else False
    }

    return comparison


def print_report(comparison_result):
    """
    Print a detailed report of the comparison result.

    Parameters:
    - comparison_result: A dictionary containing the comparison results.

    Returns:
    - None. Prints the results to the console.
    """
    print("=== Pyg Object Comparison Report ===\n")

    for comparison_key, result in comparison_result.items():
        print(f"--- {comparison_key} ---")

        # Report the number of nodes
        num_nodes = result['num_nodes']
        print(f"Number of Nodes - {comparison_key}:")
        print(f"  {comparison_key.split(' vs ')[0]}: {num_nodes[comparison_key.split(' vs ')[0]]}")
        print(f"  {comparison_key.split(' vs ')[1]}: {num_nodes[comparison_key.split(' vs ')[1]]}")
        print(f"  Match: {num_nodes['match']}\n")

        # Report the number of edges
        num_edges = result['num_edges']
        print(f"Number of Edges - {comparison_key}:")
        print(f"  {comparison_key.split(' vs ')[0]}: {num_edges[comparison_key.split(' vs ')[0]]}")
        print(f"  {comparison_key.split(' vs ')[1]}: {num_edges[comparison_key.split(' vs ')[1]]}")
        print(f"  Match: {num_edges['match']}\n")

        # Report whether node features match
        node_features = result['node_features_match']
        print(f"Node Features Match: {node_features['match']}\n")

        # Report whether labels match (if present)
        labels_match = result['labels_match']
        print(f"Labels Match: {labels_match['match']}\n")

    print("=== End of Report ===\n")

def load_norman_subset(data_dir):
    pert_data = PertData(data_dir)
    pert_data.load(data_path=data_dir)
    pert_data.dataset_name = "norman"
    pert_data.prepare_split(split='no_test', seed=42)
    adata = pert_data.adata
    adata.uns['subgroups'] = pert_data.subgroup
    save_split_data(adata, data_dir, "perturb_processed.h5ad")
    train = adata[adata.obs.split == "train"].copy()
    val = adata[adata.obs.split == "val"].copy()
    # No need to save train_val since the subset is already exist in reverse folder
    return train, val

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = os.path.join(args.data_dir, args.dataset)
    if args.reverse:
        data_dir = os.path.join(data_dir, "reverse")
    split_dir = f"{data_dir}/{args.n_clients}_clients"
    if not os.path.exists(split_dir):
        os.makedirs(split_dir, exist_ok=True)
    if args.reverse:
        assert args.dataset == "norman", "Reverse splitting is only supported for the Norman dataset."
        train_adata, val_adata = load_norman_subset(data_dir)
    else:
        train_adata, val_adata = read_split(args.data_dir, args.dataset)
    clients_data = split_by_condition(train_adata, val_adata, args.n_clients, split_dir, args.train_filename)
    plot_distribution(clients_data, split_dir)