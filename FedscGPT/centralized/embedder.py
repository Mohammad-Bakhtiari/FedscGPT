import copy
import os
from tqdm import tqdm
import torch
import scgpt as scg
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from scgpt.data_collator import DataCollator
from scgpt.tasks.cell_emb import load_pretrained
from FedscGPT.centralized.models import ScGPT
from FedscGPT.utils import Dataset, read_h5ad, get_similar_vectors

try:
    import faiss
    faiss_imported = True
except:
    faiss_imported = False
    print("Faiss not installed. Using numpy for similarity search.")


class Embedder(ScGPT):
    def __init__(self, data_dir, reference_adata, pretrained_model_dir, gene_col, k, **kwargs):
        super().__init__(data_dir, pretrained_model_dir, **kwargs)
        self.gene_col = gene_col
        self.read_reference(reference_adata)
        self.load_pretrained_config(set_pretrained_config=True)
        self.embed_adata = self.filter_id_in_vocab(self.adata)
        self.vocab.set_default_index(self.vocab["<pad>"])
        if self.gene_col == 'index':
            self.embed_adata.var[self.gene_col] = self.embed_adata.var.index
        self.gene_ids = np.array(self.vocab(self.embed_adata.var[self.gene_col].tolist()), dtype=int)
        self.instantiate_transformer_model()
        load_pretrained(self.model, torch.load(pretrained_model_dir + "/best_model.pt"), verbose=self.verbose)
        self.model.to(self.device)
        self.model.eval()
        assert self.config.model.cell_emb_style == "cls", f"Unknown cell embedding mode: {self.config.model.cell_emb_style}"
        self.max_length = 1200
        self.n_samples = None
        self.k = k

    def filter_id_in_vocab(self, adata):
        adata.var["id_in_vocab"] = [
            self.vocab[gene] if gene in self.vocab else -1 for gene in adata.var[self.gene_col]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        self.log(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(self.vocab)}."
        )
        return adata[:, adata.var["id_in_vocab"] >= 0]


    def load_dataloader(self, adata):
        count_matrix = adata.X
        count_matrix = (
            count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A
        )
        batch_ids = np.array(adata.obs["batch_id"].tolist()) if self.config.model.use_batch_labels else None
        dataset = Dataset(self.vocab, count_matrix, self.gene_ids, self.config.preprocess.cell_emb_style, self.config.preprocess.pad_value, batch_ids)
        self.n_samples = len(dataset)
        collator = DataCollator(
            do_padding=True,
            pad_token_id=self.vocab[self.config.preprocess.pad_token],
            pad_value=self.config.preprocess.pad_value,
            do_mlm=False,
            do_binning=True,
            max_length=self.max_length,
            sampling=True,
            keep_first_n_tokens=1,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.train.batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), self.config.train.batch_size),
            pin_memory=True,
        )
        return data_loader


    def embed_adata_file(self, adata=None):
        if adata is None:
            adata = self.embed_adata
        model_config = copy.deepcopy(self.config.model.__dict__)
        model_config["pad_token"] = self.config.preprocess.pad_token
        model_config["pad_value"] = self.config.preprocess.pad_value
        data_loader = self.load_dataloader(adata)
        cell_embeddings = self.get_cell_embeddings(data_loader)
        adata.obsm["X_scGPT"] = cell_embeddings
        return adata


    def get_cell_embeddings(self, data_loader):
        cell_embeddings = np.zeros((self.n_samples, self.config.model.embsize), dtype=np.float32)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            count = 0
            for data_dict in tqdm(data_loader, desc="Embedding cells"):
                input_gene_ids = data_dict["gene"].to(self.device)
                src_key_padding_mask = input_gene_ids.eq(
                    self.vocab[self.config.preprocess.pad_token]
                )
                embeddings = self.model._encode(
                    input_gene_ids,
                    data_dict["expr"].to(self.device),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=data_dict["batch_labels"].to(self.device)
                    if self.config.model.use_batch_labels
                    else None,
                )

                embeddings = embeddings[:, 0, :]  # get the <cls> position embedding
                embeddings = embeddings.cpu().numpy()
                cell_embeddings[count: count + len(embeddings)] = embeddings
                count += len(embeddings)
        cell_embeddings = cell_embeddings / np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
        return cell_embeddings

    def embed_query_adata(self, data_dir, adata):
        self.log(f"Reading query data from {adata}")
        adata = read_h5ad(data_dir, adata)
        embed_adata = self.filter_id_in_vocab(adata)
        adata_embed = self.embed_adata_file(embed_adata)
        return adata, adata_embed

    def reference_map(self, query, embed_query):
        index = faiss.IndexFlatL2(self.embed_adata.obsm["X_scGPT"].shape[1])
        index.add(self.embed_adata.obsm["X_scGPT"])

        # Query dataset, k - number of closest elements (returns 2 numpy arrays)
        distances, labels = index.search(embed_query.obsm["X_scGPT"], self.k)
        idx_list = [i for i in range(embed_query.obsm["X_scGPT"].shape[0])]
        preds = []
        for k in idx_list:
            if faiss_imported:
                idx = labels[k]
            else:
                idx, sim = get_similar_vectors(embed_query.obsm["X_scGPT"][k][np.newaxis, ...], self.embed_adata.obsm["X_scGPT"], k)
            pred = self.embed_adata.obs[self.celltype_key][idx].value_counts()
            preds.append(pred.index[0])
        gt = query.obs[self.celltype_key].to_numpy()
        return gt, preds




