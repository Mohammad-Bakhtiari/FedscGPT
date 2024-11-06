import __init__
from FedscGPT.centralized.PerturbationPredictor import Training, Inference, read_train_data
from FedscGPT.federated.PerturbationPredictor import FedPerturbationPredictorTraining
from FedscGPT.utils import (dump_perturbation_results, dump_pert_subgroup_results, load_and_plot_perturbation_results,
                            aggregate, GPUUsageTracker)
from args import instantiate_args, add_perturbation_args, add_federated_perturbation_args

def centralized_finetune(**kwargs):
    model = Training(**kwargs)
    model.setup()
    model.setup_losses()
    model.train_and_validate()
    model.save_best_model()
    return model.best_model.state_dict()


import anndata
def centralized_inference(inference_model=None, weights=None, dump_results=True, exp_params=None, **kwargs):
    if exp_params is None:
        exp_params = dict()
    if inference_model is None:
        inference_model = Inference(**kwargs)
        inference_model.setup()
    elif not inference_model.model_on_gpu:
        inference_model.move_model_to_gpu()
    if weights is None:
        inference_model.log("Inferencing cell types using initial weights!")
    else:
        inference_model.model.load_state_dict(weights)
    results = []
    if kwargs["reverse"]:
        test_gene, test_cond, train_cond, val_cond = inference_model.get_conditions_list()
        if dump_results:
            inference_model.plot_condition_matrix(test_gene, test_cond, train_cond, val_cond)
        inference_model.evaluate_gene_interactions(test_gene, test_cond, exp_params)
    else:
        perturbation_analysis(dump_results, inference_model, kwargs, exp_params, results)
    inference_model.move_model_to_cpu()
    return inference_model


def perturbation_analysis(dump_results, inference_model, kwargs, exp_params, results):
    for pert in kwargs["perts_to_plot"]:
        res = inference_model.predict_perturbation_outcome(pert)
        if dump_results:
            load_and_plot_perturbation_results(*res, plot_filename=f"{kwargs['output_dir']}/{pert}.png")
        results.append([pert, res])
    if dump_results:
        dump_perturbation_results(results,
                                  pool_size=inference_model.config.log.pool_size,
                                  save_file=f"{kwargs['output_dir']}/perturbation_results.pkl")
    # evaluation using entire dataset including ctrl in train....
    test_metrics, test_res = inference_model.evaluate()
    # deeper analysis on the entire dataset
    # only applicable for research purpose with serious privacy concerns
    deeper_res, non_dropout_res = inference_model.deeper_analysis(test_res)
    analysis = inference_model.subgroup_analysis(deeper_res, non_dropout_res)
    if dump_results:
        dump_pert_subgroup_results(test_metrics, analysis, save_file=f"{kwargs['output_dir']}/deeper_analysis.pkl")
    if exp_params:
        inference_model.record_results(exp_params, test_metrics, analysis)



def centralized_finetune_inference(**kwargs):
    weights = centralized_finetune(**kwargs)
    centralized_inference(weights=weights, **kwargs)


def federated_finetune(per_round_eval, **kwargs):
    fed_model = FedPerturbationPredictorTraining(**kwargs)
    fed_model.init_global_weights()
    if per_round_eval:
        inference_model = centralized_inference(weights=fed_model.global_weights,
                                                dump_results=False,
                                                exp_params={"round": 0, 'epoch': kwargs['n_epochs']},
                                                **kwargs)
    n_local_samples = [client.n_samples for client in fed_model.clients]
    for round in range(1, fed_model.fed_config.n_rounds + 1):
        fed_model.logger.federated(f"Round {round}")
        local_weights = fed_model.update_clients_model(round_num=round)
        aggregate(fed_model, local_weights, n_local_samples)
        if per_round_eval:
            last_round = round == fed_model.fed_config.n_rounds
            centralized_inference(inference_model=inference_model,
                                  weights=fed_model.global_weights,
                                  dump_results=last_round,
                                  exp_params={"round": round, 'epoch': kwargs['n_epochs']},
                                  **kwargs)
        if fed_model.stop():
            break
    fed_model.delete_clients_models()
    if per_round_eval:
        inference_model.save_records()
    else:
        centralized_inference(weights=fed_model.global_weights, exp_params={"round": round, 'epoch': kwargs['n_epochs']},
                              **kwargs)
    fed_model.save_global_weights()



def clients_centralized_training(**kwargs):
    fed_model = FedPerturbationPredictorTraining(**kwargs)
    kwargs.pop('output_dir')
    inference_model = None
    for client in fed_model.clients:
        client.move_model_to_gpu()
        client.train_and_validate()
        client.move_model_to_cpu()
        inference_model = centralized_inference(inference_model=inference_model,
                                                weights=client.best_model.state_dict(),
                                                output_dir=client.output_dir,
                                                **kwargs)


if __name__ == '__main__':
    parser = instantiate_args()
    add_perturbation_args(parser)
    add_federated_perturbation_args(parser)
    args = parser.parse_args()
    args.perts_to_plot = args.perts_to_plot.split(',')

    if args.verbose:
        tracker = GPUUsageTracker()


    if args.mode == 'centralized':
        centralized_finetune_inference(task="perturbation", **vars(args))
    elif args.mode == 'centralized_inference':
        args.init_weights_dir = f"{args.output_dir}/model/model.pt"
        centralized_inference(task="perturbation", **vars(args))
    elif args.mode == 'federated_finetune':
        federated_finetune(task="perturbation", **vars(args))
    elif args.mode == 'centralized_clients':
        clients_centralized_training(task="perturbation", **vars(args))


    if args.verbose:
        tracker.generate_report()
