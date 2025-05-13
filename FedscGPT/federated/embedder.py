import faiss
import hashlib
import numpy as np
import torch
import crypten
from FedscGPT.base import FedBase
from FedscGPT.utils import read_h5ad, top_k_encrypted_distances, top_k_ind_selection
from FedscGPT.centralized.embedder import Embedder


class ClientEmbedder(Embedder):
    """
    Client-side embedder extending centralized Embedder to support
    local distance computation, secure multiparty computation (SMPC),
    and privacy-preserving hashing for federated nearest-neighbor voting.
    """

    def __init__(self, smpc=False, **kwargs):
        """
        Initialize the ClientEmbedder.

        Parameters
        ----------
        smpc : bool, optional
            If True, enable secure multiparty computation mode.
        **kwargs
            Additional keyword arguments forwarded to the base Embedder.
        """
        super().__init__(**kwargs)
        self.ind_offset = None
        self.mapped_ct = None
        self.enc_celltype_ind_offset = None
        self.celltype_ind_offset = None
        self.label_to_index = None
        self.smpc = smpc
        self.hash_index_map = {}
        self.embed_adata = self.embed_adata_file()
        del self.model

    def compute_local_distances(self, query_embeddings):
        """
        Compute distances between query embeddings and this client's local reference embeddings.

        Supports both non-SMPC (plaintext) and SMPC modes.

        Parameters
        ----------
        query_embeddings : np.ndarray or crypten.CrypTensor
            Query embeddings of shape (n_queries, embedding_dim).  In non-SMPC mode, a NumPy array;
            in SMPC mode, a CrypTensor.

        Returns
        -------
        distances : np.ndarray or crypten.CrypTensor
            Top-k distances per query, shape (n_queries, k). Plain float array if non-SMPC;
            encrypted CrypTensor if SMPC.
        hashed_indices : list of list of str
            Hashed reference indices for each query sample, preserving privacy.
        """
        if self.smpc:
            return self.compute_squared_distances(query_embeddings)

        dim = self.embed_adata.obsm["X_scGPT"].shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(self.embed_adata.obsm["X_scGPT"])
        distances, indices = index.search(query_embeddings, self.k)
        hashed_indices = self.hash_indices(indices)
        return distances, hashed_indices

    def compute_squared_distances(self, secure_embeddings):
        """
        Compute encrypted squared L2 distances between secure query embeddings and local reference.

        Parameters
        ----------
        secure_embeddings : crypten.CrypTensor
            Encrypted query embeddings, shape (n_queries, embedding_dim).

        Returns
        -------
        encrypted_topk : crypten.CrypTensor
            Encrypted top-k distances per query, shape (n_queries, k).
        hashed_indices : crypten.CrypTensor
            Encrypted, mapped reference indices for top-k neighbors, shape (n_queries, k).
        """
        reference = torch.tensor(
            self.embed_adata.obsm["X_scGPT"], dtype=torch.float32, device=self.device
        )
        reference = crypten.cryptensor(reference)

        # Compute norms and cross-terms
        query_norm = secure_embeddings.square().sum(dim=1).unsqueeze(1)
        ref_norm = reference.square().sum(dim=1).unsqueeze(0)
        cross = secure_embeddings @ reference.transpose(0, 1)

        distances = query_norm + ref_norm - 2 * cross
        del reference, query_norm, ref_norm, cross

        encrypted_topk, topk_indices = top_k_encrypted_distances(distances, self.k)
        mapped = [(
            idx @ self.enc_celltype_ind_offset.unsqueeze(1)
        ).squeeze(1) for idx in topk_indices]
        hashed_indices = crypten.stack(mapped, dim=1)
        return encrypted_topk, hashed_indices

    def hash_indices(self, indices):
        """
        Hash local reference indices using a client-specific salt for privacy.

        Parameters
        ----------
        indices : np.ndarray
            Array of local reference indices to hash, shape (n_queries, k).

        Returns
        -------
        list of list of str
            Nested list of SHA-256 hexadecimal hashes for each index.
        """
        salt = self.get_client_salt()
        hashed = []
        for row in indices:
            row_hashes = []
            for idx in row:
                h = hashlib.sha256(f"{salt}-{idx}".encode()).hexdigest()
                row_hashes.append(h)
                self.hash_index_map[h] = idx
            hashed.append(row_hashes)
        return hashed

    def get_client_salt(self):
        """
        Generate a unique client-specific salt for hashing indices.

        Returns
        -------
        str
            String representation of the client's log identifier.
        """
        return str(self.log_id)

    def vote(self, global_nearest_samples):
        """
        Perform voting on global nearest neighbor samples to produce local votes.

        In non-SMPC mode, returns a count dict per query. In SMPC mode, returns encrypted vote vectors.

        Parameters
        ----------
        global_nearest_samples : list of list of str or crypten.CrypTensor
            Hashed global nearest neighbor indices per query, shape (n_queries, k).

        Returns
        -------
        votes : list of dict or list of crypten.CrypTensor
            If non-SMPC: list of dicts mapping label->vote count for each query.
            If SMPC: list of encrypted vote vectors (CrypTensors) per query.
        """
        votes = []
        if self.smpc:
            n_q = global_nearest_samples.size(0)
            local_inds = self.enc_celltype_ind_offset.unsqueeze(0)
            ct_enc = crypten.cryptensor(
                torch.tensor(self.mapped_ct, dtype=torch.long, device=self.device)
            )
            ct_exp = ct_enc.unsqueeze(0).expand(n_q, self.n_samples)
            for i in range(self.k):
                sample_k = (
                    global_nearest_samples[:, i].unsqueeze(1)
                    .expand(n_q, self.n_samples)
                )
                mask = (sample_k == local_inds)
                votes.append((mask * ct_exp).sum(dim=1).unsqueeze(1))
            return crypten.cat(votes, dim=1)
        else:
            for row in global_nearest_samples:
                counts = {}
                for h in row:
                    if h in self.hash_index_map:
                        idx = self.hash_index_map[h]
                        label = self.adata.obs[self.celltype_key].values[idx]
                        counts[label] = counts.get(label, 0) + 1
                votes.append(counts)
            return votes

    def report_celltypes(self):
        """
        Report unique cell type labels in this client's reference data.

        Returns
        -------
        set of str
            Unique cell type labels.
        """
        return set(self.adata.obs[self.celltype_key].unique())

    def harmonize_celltypes(self, global_label_to_index, ind_offset):
        """
        Map local cell types to global indices and optionally encrypt offsets for SMPC.

        Parameters
        ----------
        global_label_to_index : dict
            Mapping from cell type label string to global integer index.
        ind_offset : int
            Offset to apply to local sample indices for global uniqueness.
        """
        self.label_to_index = global_label_to_index
        labels = self.embed_adata.obs[self.celltype_key].values
        self.mapped_ct = [global_label_to_index[ct] for ct in labels]
        self.ind_offset = ind_offset
        if self.smpc:
            indices = torch.arange(self.n_samples, dtype=torch.long, device=self.device) + ind_offset
            self.enc_celltype_ind_offset = crypten.cryptensor(indices)

    def report_n_local_samples(self):
        """
        Report the number of local reference samples as an encrypted tensor.

        Returns
        -------
        crypten.CrypTensor
            Encrypted count of local samples.
        """
        count = torch.tensor(self.n_samples, dtype=torch.float32, device=self.device)
        return crypten.cryptensor(count)


class FedEmbedder(FedBase):
    """
    Federated embedder coordinating multiple ClientEmbedders to perform
    secure or plaintext federated nearest-neighbor classification.
    """

    def __init__(
        self, data_dir, reference_adata, query_adata, output_dir, k, smpc=False, **kwargs
    ):
        """
        Initialize the federated embedder and distribute reference data to clients.

        Parameters
        ----------
        data_dir : str
            Directory containing H5AD files.
        reference_adata : str
            Filename of the reference AnnData file.
        query_adata : str
            Filename of the query AnnData file.
        output_dir : str
            Directory to write outputs.
        k : int
            Number of nearest neighbors to consider.
        smpc : bool, optional
            If True, enable secure multiparty computation.
        **kwargs
            Additional parameters, including 'batch_key' and 'celltype_key'.
        """
        super().__init__(data_dir=data_dir, output_dir=output_dir, **kwargs)
        self.total_n_samples = None
        self.label_to_index = None
        self.index_to_label = None
        self.smpc = smpc
        self.k = k
        self.n_classes = None

        adata = read_h5ad(data_dir, reference_adata)
        self.distribute_adata_by_batch(adata, kwargs['batch_key'], keep_vars=True)
        self.celltype_key = kwargs['celltype_key']

        # Instantiate client embedders
        for c in range(self.n_clients):
            client = ClientEmbedder(
                reference_adata='adata.h5ad',
                data_dir=self.clients_data_dir[c],
                output_dir=self.clients_output_dir[c],
                log_id=f"client_{self.client_ids[c]}",
                k=self.k,
                logger=self.logger,
                smpc=smpc,
                **kwargs
            )
            self.clients.append(client)

        # Embed query data
        self.query, self.embed_query = self.embed_query_adata(
            query_adata,
            output_dir=output_dir,
            data_dir=data_dir,
            **kwargs
        )
        self.n_query_samples = self.query.shape[0]
        if self.smpc:
            arr = torch.tensor(
                self.embed_query.obsm["X_scGPT"], dtype=torch.float32, device=self.device
            )
            self.embed_query.obsm['secure_embed'] = crypten.cryptensor(arr)

    def embed_query_adata(self, query_adata, **kwargs):
        """
        Embed the query AnnData using the centralized Embedder logic.

        Parameters
        ----------
        query_adata : str
            Filename of the query AnnData file.
        **kwargs
            Additional parameters passed to Embedder.

        Returns
        -------
        tuple
            - AnnData with original query data.
            - Embedder object containing embedded data.
        """
        embedder = Embedder(reference_adata=query_adata, k=self.k, **kwargs)
        embedder.embed_adata_file()
        del embedder.model
        return embedder.adata, embedder.embed_adata

    def global_aggregate_distances(self, client_distances, client_indices):
        """
        Aggregate distances across clients to determine global top-k nearest neighbors.

        Parameters
        ----------
        client_distances : list of np.ndarray or crypten.CrypTensor
            Distances from each client, shape per-client (n_queries, k).
        client_indices : list of list of list of str or crypten.CrypTensor
            Hashed neighbor indices from each client.

        Returns
        -------
        list of list of str or crypten.CrypTensor
            Global top-k hashed reference indices per query.
        """
        if self.smpc:
            return self.secure_top_k_distance_agg(client_distances, client_indices)

        all_dist = np.hstack(client_distances)
        all_hash = np.hstack(client_indices)
        order = np.argsort(all_dist, axis=1)
        result = []
        for i, row in enumerate(order):
            result.append([all_hash[i, j] for j in row[:self.k]])
        return result

    def secure_top_k_distance_agg(self, client_distances, client_indices):
        """
        Securely aggregate encrypted distances and indices across clients.

        Parameters
        ----------
        client_distances : list of crypten.CrypTensor
            Encrypted local top-k distance matrices from clients.
        client_indices : list of crypten.CrypTensor
            Encrypted hashed index tensors from clients.

        Returns
        -------
        crypten.CrypTensor
            Encrypted global top-k hashed reference indices, shape (n_queries, k).
        """
        distances = crypten.cat(client_distances, dim=1)
        indices = crypten.cat(client_indices, dim=1)
        one_hot = top_k_ind_selection(distances.clone(), self.k)
        top_k = [(one_hot[:, i:i+1] * indices).sum(dim=1, keepdim=True)
                 for i in range(self.k)]
        return crypten.cat(top_k, dim=1)

    def aggregate_client_votes(self, client_votes):
        """
        Aggregate vote counts from all clients to produce final predictions.

        Parameters
        ----------
        client_votes : list of list of dict or list of crypten.CrypTensor
            Votes returned from each client.

        Returns
        -------
        np.ndarray
            Predicted labels for each query sample.
        """
        if self.smpc:
            stacked = crypten.stack(client_votes, dim=2).sum(dim=2)
            flat = stacked.view(-1, 1)
            classes = torch.arange(self.n_classes, device=self.device).view(1, -1)
            one_hot = (flat == classes).view(self.n_query_samples, self.k, self.n_classes)
            _, argmax = one_hot.sum(dim=1).max(dim=1)
            preds = argmax.get_plain_text().cpu().numpy().astype(int)
            return np.array([self.index_to_label[p] for p in preds], dtype=object)

        # Plaintext aggregation
        agg = [{} for _ in range(self.n_query_samples)]
        for votes in client_votes:
            for i, vote_dict in enumerate(votes):
                for lbl, cnt in vote_dict.items():
                    agg[i][lbl] = agg[i].get(lbl, 0) + cnt
        preds = []
        for counts in agg:
            if counts:
                preds.append(max(counts, key=counts.get))
            else:
                preds.append(None)
        return np.array(preds, dtype=object)

    def federated_reference_map(self):
        """
        Execute the full federated mapping workflow: distance computation, aggregation,
        voting, and prediction.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            - Ground truth labels of query samples.
            - Predicted labels from federated vote.
        """
        self.collect_global_celltypes()
        cdists, cinds = [], []
        key = 'secure_embed' if self.smpc else 'X_scGPT'
        query_emb = self.embed_query.obsm[key]
        for client in self.clients:
            d, i = client.compute_local_distances(query_emb)
            cdists.append(d)
            cinds.append(i)
        global_inds = self.global_aggregate_distances(cdists, cinds)
        votes = [client.vote(global_inds) for client in self.clients]
        preds = self.aggregate_client_votes(votes)
        gt = self.query.obs[self.celltype_key].to_numpy()
        return gt, preds

    def collect_global_celltypes(self):
        """
        Build global label mappings from all clients' cell types and harmonize clients.
        """
        labels = set()
        for client in self.clients:
            labels |= client.report_celltypes()
        sorted_lbls = sorted(labels)
        self.label_to_index = {lbl: idx for idx, lbl in enumerate(sorted_lbls, 1)}
        self.index_to_label = {idx: lbl for lbl, idx in self.label_to_index.items()}
        self.n_classes = len(self.label_to_index)
        if self.smpc:
            self.aggregate_total_n_samples()
            offset = 0
            for client in self.clients:
                client.harmonize_celltypes(self.label_to_index, offset)
                offset += self.total_n_samples
        else:
            for client in self.clients:
                client.harmonize_celltypes(self.label_to_index, 0)

    def aggregate_total_n_samples(self):
        """
        Compute the total number of samples across all clients in encrypted form.

        Sets
        ----
        total_n_samples : int
            Plaintext total number of local samples summed across clients.
        """
        counts = [client.report_n_local_samples() for client in self.clients]
        total = counts[0]
        for c in counts[1:]:
            total += c
        self.total_n_samples = int(total.get_plain_text().item())
