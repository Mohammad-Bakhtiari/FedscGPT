""""

"""
import __init__
from FedscGPT.utils import confusion_matrix_evaluation, split_data_by_batch, save_data_batches, set_seed
from FedscGPT.centralized.annotator import CellTypeAnnotator, Training, Inference
from FedscGPT.federated.annotator import FedAnnotator
from FedscGPT.federated.aggregator import FedAvg
import os
import torch
from functools import partial


def centralized_finetune(**kwargs):
    """
    We assume there is a test_set that should be combined with the training set for preprocessing and then split
    after preprocessing.

    Parameters
    ----------
    query_adata
    reference_adata
    config
    data_dir
    adata
    test_adata
    pretrained_model_dir
    output_dir

    Returns
    -------

    """
    annotator = cent_prep(**kwargs)
    annotator.post_prep()
    annotator.tokenize()
    annotator.instantiate_transformer_model()
    annotator.load_pretrained_model()
    annotator.setup_losses()
    annotator.train()
    annotator.save_best_model(annotator.model)
    return annotator


def cent_prep(annotator=None, preprocess=True, **kwargs):
    if annotator is None:
        annotator = Training(**kwargs)
    annotator.harmonize(annotator.adata)
    if annotator.pretrained_model_dir is not None:
        annotator.load_pretrained_config()
    annotator.adata = annotator.filter(annotator.adata)
    if preprocess:
        annotator.instantiate_preprocessor()
        annotator.preprocess_reference()
    return annotator

def centralized_finetune_inference(**kwargs):
    """
    """
    annotator = centralized_finetune(**kwargs)
    inference_model = centralized_inference(logger=annotator.logger,
                                            load_model=False,
                                            weights=annotator.best_model.state_dict(),
                                            plot_results=True,
                                            **kwargs
                                            )
    inference_model.save_records()

def centralized_inference(annotator=None,
                          logger=None,
                          load_model=True,
                          weights=None,
                          plot_results=False,
                          model_name='model.pt',
                          round_number=None,
                          **kwargs,
                          ):
    instantiate = False
    if annotator is None:
        annotator = Inference(log_id="inference", logger=logger, load_model=load_model, model_name=model_name, **kwargs)
        instantiate = True
    if not load_model:
        if weights is None:
            raise Warning("Inferencing cell types using random network!")
        else:
            annotator.best_model.load_state_dict(weights)
    if (not instantiate) or (round_number is not None):
        predictions, labels = annotator.inference(plot_results=plot_results,
                                                  round_num=round_number,
                                                  n_epochs=kwargs['n_epochs'],
                                                  mu=kwargs['mu'] if kwargs['use_fedprox'] else None)

    if plot_results:
        confusion_matrix_evaluation(annotator.unique_cell_types,
                                    predictions,
                                    labels,
                                    annotator.cell_id2type,
                        f"{kwargs['output_dir']}/plots",

                                    )
    return annotator

from FedscGPT.utils import list_gpu_objects

def federated_finetune(**kwargs):
    annotator = fed_prep(**kwargs)
    annotator.post_prep_setup()
    annotator.init_global_weights()
    cent_inf = partial(centralized_inference, agg_method="federated", logger=annotator.logger, load_model=False, **kwargs)
    # inference_model = cent_inf(weights=annotator.global_weights, save_results=False, round_number=0)
    n_local_samples = [client.n_samples for client in annotator.clients]
    for round in range(1, annotator.fed_config.n_rounds + 1):
        annotator.logger.federated(f"Round {round}")
        local_weights = annotator.update_clients_model(round_num=round)
        annotator.aggregate(local_weights, n_local_samples=n_local_samples)
        last_round = round == annotator.fed_config.n_rounds
    #     cent_inf(annotator=inference_model, weights=annotator.global_weights, plot_results=last_round, round_number=round)
    #     if annotator.stop():
    #         break
    # inference_model.save_records()

def fed_prep(annotator=None, harmonize=False, **kwargs):
    if annotator is None:
        annotator = FedAnnotator(**kwargs)
    if harmonize:
        annotator.aggregate_gene_sets()
        annotator.aggregate_celltype_sets()
    else:
        for client in annotator.clients:
            client.local_harmonize()
    annotator.load_pretrained_config()
    annotator.filter_genes()
    annotator.preprocess_data()
    return annotator


def centralized_prep_fed_annotation(reference_adata, **kwargs):
    annotator = cent_prep(reference_adata=reference_adata, **kwargs)
    temp_ref_file_path = "/".join(reference_adata.split("/")[:-1] + ["prep_ref.h5ad"])
    annotator.adata.var.drop(columns=["gene_name"], inplace=True)
    annotator.adata.write_h5ad(temp_ref_file_path, compression="gzip")
    federated_finetune(reference_adata=temp_ref_file_path, **kwargs)


def clients_centralized_training(federated_prep, **kwargs):
    if federated_prep:
        print("Federated preparation")
        fed_annotator = fed_prep(**kwargs)
        fed_annotator.save_clients_fed_prep_data()
        client_ref_adata = "fed_prep_adata.h5ad"
        clients_data_dir = fed_annotator.clients_data_dir
        client_output_dir = fed_annotator.clients_output_dir
    else:
        client_ref_adata = "cent_prep_adata.h5ad"
        cent_annotator = cent_prep(**kwargs)
        batches = split_data_by_batch(cent_annotator.adata, kwargs['batch_key'], keep_vars=False)
        clients_data_dir = save_data_batches(batches, kwargs['data_dir'], filename=client_ref_adata)
        client_output_dir = [f"{kwargs['output_dir']}/{d.split('/')[-1]}" for d in clients_data_dir]
    for data_dir, output_dir in zip(clients_data_dir, client_output_dir):
        kwargs['reference_adata'] = client_ref_adata
        kwargs['data_dir'] = data_dir
        kwargs['output_dir'] = output_dir
        centralized_finetune_inference(preprocess=False, agg_method=os.path.basename(data_dir), **kwargs)

def single_shot_fed(**kwargs):
    cent_inf = partial(centralized_inference, load_model=False, save_results=False, **kwargs)
    models = []
    model_dir = kwargs["output_dir"].replace("federated", "centralized")
    for d in os.listdir(model_dir):
        if "client_" in d:
            models.append(torch.load(f"{model_dir}/{d}/model/model.pt"))

    inference_model, results = cent_inf(weights=models[0])
    for model in models[1:]:
        inference_model, results = cent_inf(annotator=inference_model, weights=model)
    agg = FedAvg(n_rounds=0)
    agg.aggregate(models)
    cent_inf(annotator=inference_model, weights=agg.global_weights)

def federated_zeroshot_annotation(config, data_dir, adata, test_adata, pretrained_model_dir, output_dir):
    raise NotImplementedError("Federated zero-shot annotation is not implemented yet.")

