import mlflow
import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.metrics import confusion_matrix


class MLflowResultsRecorder:
    def __init__(self, dataset, run_name=None, experiment_name="FedscGPT", agg_method="FedAvg", verbose=False):
        self.dataset = dataset
        self.agg_method = agg_method
        self.verbose = verbose

        self.all_results = {}
        self.all_results[dataset] = {agg_method: {}}

        # Ensure output directory under ~/FedscGPT/output
        self.output_dir = os.path.expanduser(f"~/FedscGPT/output/{dataset}/{agg_method}/{run_name or 'default'}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Configure MLflow
        mlruns_dir = os.path.expanduser("~/FedscGPT/output/mlruns")
        mlflow.set_tracking_uri(f"file:{mlruns_dir}")
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)

        mlflow.log_param("dataset", dataset)
        mlflow.log_param("aggregation_method", agg_method)

        self.pickle_file = os.path.join(self.output_dir, f"{run_name or 'results'}_details.pkl")

    def update(self, accuracy, precision, recall, macro_f1, predictions, labels, id_maps, round_number, n_epochs, mu=None):
        self.record_metrics(round_number, accuracy, precision, recall, macro_f1, n_epochs, mu)
        self.record_detailed_results(predictions, labels, id_maps, round_number, n_epochs, mu)

    def record_metrics(self, round_number, accuracy, precision, recall, macro_f1, n_epochs=None, mu=None):
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "macro_f1": macro_f1
        }, step=round_number)

        if self.verbose:
            print(f"[Round {round_number}] Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {macro_f1:.3f}")

        if n_epochs is not None:
            mlflow.log_param(f"n_epochs_round_{round_number}", n_epochs)
        if mu is not None:
            mlflow.log_param(f"mu_round_{round_number}", mu)

    def record_detailed_results(self, predictions, labels, id_maps, round_number, epoch, mu=None):
        dataset = self.dataset
        agg = self.agg_method

        if 'id_maps' not in self.all_results[dataset]:
            self.all_results[dataset]['id_maps'] = id_maps
        else:
            assert self.all_results[dataset]['id_maps'] == id_maps, f"ID Maps mismatch for dataset {dataset}"

        self.all_results[dataset].setdefault(agg, {})
        self.all_results[dataset][agg].setdefault(epoch, {})
        self.all_results[dataset][agg][epoch].setdefault(round_number, {})
        self.all_results[dataset][agg][epoch][round_number][mu] = {
            'predictions': predictions,
            'labels': labels
        }

        # Save confusion matrix (CSV only)
        self.log_confusion_matrix(predictions, labels, id_maps['id2type'], id_maps['celltypes'], round_number, epoch, mu)

    def save_detailed_results_pickle(self):
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(self.all_results, f)
        mlflow.log_artifact(self.pickle_file, artifact_path="results")
        if self.verbose:
            print(f"Saved detailed results to {self.pickle_file}")

    def end_run(self):
        self.save_detailed_results_pickle()
        mlflow.end_run()

    def log_confusion_matrix(self, predictions, labels, id2type: Dict[int, str], class_names: List[str], round_number=None, epoch=None, mu=None):
        # Map predicted and true labels to their string names
        pred_names = [id2type[p] for p in predictions]
        label_names = [id2type[l] for l in labels]

        # Ensure valid class names
        valid_classes = list({*pred_names, *label_names})
        if class_names:
            valid_classes = [cls for cls in class_names if cls in valid_classes]

        cm = confusion_matrix(label_names, pred_names, labels=valid_classes)
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = cm / np.where(row_sums == 0, 1, row_sums)

        cm_df = pd.DataFrame(cm_normalized, index=valid_classes, columns=valid_classes)

        artifact_name = f"confusion_matrix_ep{epoch}_r{round_number}_mu{mu or 0}.csv"
        cm_path = os.path.join(self.output_dir, artifact_name)
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(cm_path)
