import __init__
from FedscGPT.centralized.embedder import Embedder
from FedscGPT.federated.embedder import FedEmbedder
from FedscGPT.utils import plot_embedding, eval_reference_mapping, set_seed
from args import instantiate_args, add_observation_args, add_federated_embedding_args




def federated_zero_shot_embedding(**kwargs):
    embedder = FedEmbedder(**kwargs)
    gt, preds = embedder.federated_reference_map()
    eval_reference_mapping(gt, preds, kwargs['output_dir'], embedder.logger.federated)



def centralized_zero_shot_embedding(data_dir, query_adata, **kwargs):
    embedder = Embedder(data_dir=data_dir, **kwargs)
    embedder.embed_adata_file()
    query, embed_query = embedder.embed_query_adata(data_dir, query_adata)
    plot_embedding(embedder.embed_adata, embed_query, cell_type_key=kwargs['celltype_key'], output_dir=kwargs['output_dir'])
    gt, preds = embedder.reference_map(query, embed_query)
    eval_reference_mapping(gt, preds, kwargs['output_dir'], embedder.log)


def clients_local_embedding(**kwargs):
    fed_embedder = FedEmbedder(**kwargs)
    print(f"Embedding query adata for {len(fed_embedder.clients)} clients.")
    for client in fed_embedder.clients:
        client.embed_adata_file()
        plot_embedding(client.embed_adata, fed_embedder.embed_query, cell_type_key=kwargs['celltype_key'],
                       output_dir=client.output_dir)
        gt, preds = client.reference_map(fed_embedder.query, fed_embedder.embed_query)
        eval_reference_mapping(gt, preds, client.output_dir, client.log)


if __name__ == '__main__':
    parser = instantiate_args()
    add_observation_args(parser)
    add_federated_embedding_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)
    print(**vars(args))
    exit()
    if args.mode == 'centralized':
        centralized_zero_shot_embedding(task="embedding", **vars(args))
    elif args.mode == 'federated_zeroshot':
        federated_zero_shot_embedding(task="embedding", **vars(args))
    elif args.mode == 'centralized_clients':
        clients_local_embedding(task="embedding", **vars(args))