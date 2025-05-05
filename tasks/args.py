import os
import argparse


def instantiate_args():
    HOME_DIR = "/home/bba1658/FedscGPT"
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['centralized', 'federated_finetune', 'federated_zeroshot',
                                                     'centralized_finetune_inference','centralized_inference', 'federated_inference',
                                                     'federated_prep', 'cent_prep_fed_finetune', 'centralized_clients', 'single_shot_federated'],
                        default='centralized')
    parser.add_argument('--data-dir', type=str, default=f"{HOME_DIR}/data/ms")
    parser.add_argument('--output-dir', type=str, default=f"{HOME_DIR}/output")
    parser.add_argument('--reference_adata', type=str, default='ms.h5ad')
    parser.add_argument('--query_adata', type=str, default='ms.h5ad')
    parser.add_argument('--pretrained_model_dir', type=str, default=f'{HOME_DIR}/pretrained_models/scGPT_human')
    parser.add_argument("--init_weights_dir", type=str, default=f"{HOME_DIR}/init_weights/hp.pth")
    parser.add_argument("--finetune_model_dir", type=str)
    parser.add_argument("--smpc", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--config_file', type=str, help='.yml file for the model', default='config.yml')
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--dataset_name", type=str, default='ms')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--param_tuning_res", type=str, default=f"{HOME_DIR}/output/annotation/param-tuning_res")
    return parser


def add_annotation_args(parser):
    parser.add_argument('--celltype_key', type=str, default='celltype')
    parser.add_argument('--batch_key', type=str, default='batch')
    parser.add_argument('--dataset', type=str, default='ms', choices=['ms', 'hp', 'myeloid'])

def add_observation_args(parser):
    parser.add_argument('--celltype_key', type=str, default='celltype')
    parser.add_argument('--batch_key', type=str, default='batch')
    parser.add_argument('--gene_col', type=str, default='gene_name')
    parser.add_argument('--k', type=int, help="Number of nearest neighbors", default=10)


def add_federated_annotation_args(parser):
    parser.add_argument('--fed_config_file', type=str, help='.yml file for the federated model',
                        default='fed_config.yml')
    parser.add_argument('--federated_prep', action='store_true', help='Enable federated preparation', default=False)
    parser.add_argument("--n_rounds", type=int, default=None, help="Number of rounds")
    parser.add_argument("--n_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--use_fedprox", action='store_true', default=False)
    parser.add_argument("--mu", type=float, default=0.05, help="Mean parameter")
    parser.add_argument("--param_tuning", action='store_true', default=False)

def add_federated_embedding_args(parser):
    parser.add_argument('--fed_config_file', type=str, help='.yml file for the federated model',
                        default='fed_config.yml')

def split_test_batches(args):
    args.test_batches = [b for b in args.test_batches.split(',')]


def create_output_dir(args):
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
