""""

"""
import __init__
import os
from args import instantiate_args, add_annotation_args, create_output_dir, add_federated_annotation_args




if __name__ == '__main__':
    parser = instantiate_args()
    add_annotation_args(parser)
    add_federated_annotation_args(parser)
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # args.gpu = 0
    from tasks import utils as annotation_utils
    annotation_utils.set_seed(args.seed)
    create_output_dir(args)
    if args.mode == "federated_finetune":
        annotation_utils.federated_finetune(task='annotation', **vars(args))
    elif args.mode == "federated_zeroshot":
        annotation_utils.federated_zeroshot_annotation(args.config_file, args.data_dir, args.adata, args.test_adata,
                                      args.pretrained_model_dir, args.output_dir)
    elif args.mode == "centralized_inference":
        annotation_utils.centralized_inference(task='annotation', agg_method="cent-inference", plot_results=True, **vars(args))
    elif args.mode == "centralized_finetune_inference":
        annotation_utils.centralized_finetune_inference(task='annotation', agg_method="centralized", **vars(args))
    elif args.mode == "cent_prep_fed_finetune":
        annotation_utils.centralized_prep_fed_annotation(task='annotation', **vars(args))
    elif args.mode == "centralized_clients":
        annotation_utils.clients_centralized_training(task='annotation', **vars(args))
    elif args.mode == "single_shot_federated":
        annotation_utils.single_shot_fed(task='annotation', **vars(args))
