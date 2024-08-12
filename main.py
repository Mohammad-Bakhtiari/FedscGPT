from scgpt.utils import set_seed

set_seed(0)

import argparse
import os
from tasks.annotation import annotate

HOME_DIR = "/home/bba1658/FedscGPT"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['annotation'], default='annotation')
    parser.add_argument('--mode', type=str, choices=['centralized', 'federated_finetuning', 'federated_zeroshot'],
                        default='centralized')
    parser.add_argument('--data-dir', type=str, default=f"{HOME_DIR}/data/ms")
    parser.add_argument('--output-dir', type=str, default=f"{HOME_DIR}/output")
    parser.add_argument('--reference_adata', type=str, default='ms.h5ad')
    parser.add_argument('--query_adata', type=str, default='ms.h5ad')
    parser.add_argument('--test_batches', type=str, default='1')
    parser.add_argument('--batch_key', type=str, default='str_batch')
    parser.add_argument('--celltype_key', type=str, default='celltype')
    parser.add_argument('--pretrained_model_dir', type=str, default=f'{HOME_DIR}/pretrained_models/scGPT_human')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--config_file', type=str, help='.yml file for the model', default='config.yml')
    parser.add_argument('--fed_config_file', type=str, help='.yml file for the federated model',
                        default='fed_config.yml')

    args = parser.parse_args()
    args.test_batches = [b for b in args.test_batches.split(',')]
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    if args.task == 'annotation':
        annotate(args.mode,
                 os.path.join(HOME_DIR, args.config_file),
                 os.path.join(HOME_DIR, args.fed_config_file),
                 args.data_dir,
                 args.adata,
                 args.batch_key,
                 args.celltype_key,
                 args.test_batches,
                 args.pretrained_model_dir,
                 args.output_dir,
                 )
