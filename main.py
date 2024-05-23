import argparse
import bios
from tasks.annotation import annotate
from FedscGPT.utils import load_config
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['annotation'], default='annotation')
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--adata', type=str, default='c_data.h5ad')
    parser.add_argument('--test_adata', type=str, default='filtered_ms_adata.h5ad')
    parser.add_argument('--dataset_name', type=str, default='ds')
    parser.add_argument('--pretrained_model_dir', type=str, default='scGPT_human')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--config_file', type=str, help='.yml file for the model')



    args = parser.parse_args()
    if args.task == 'annotation':
        annotate(os.path.join(os.getcwd(), "FedscGPT", args.config_file),
                 args.data_dir,
                 args.adata,
                 args.test_adata,
                 args.pretrained_model_dir,
                 args.output_dir)
