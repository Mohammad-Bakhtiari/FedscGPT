import scgpt as scg
import faiss
import hashlib
import numpy as np
import torch
import crypten
from FedscGPT.base import FedBase, BaseMixin
from FedscGPT.utils import (read_h5ad, concat_encrypted_distances, top_k_encrypted_distances,
                            get_plain_indices, top_k_ind_selection, encrypted_present_hashes, save_data_batches)
from FedscGPT.centralized.embedder import Embedder


class ClientEmbedder(Embedder):
    def __init__(self, smpc=False, **kwargs):
        super().__init__(**kwargs)
        self.enc_celltype_ind_offset = None
        self.celltype_ind_offset = None
        self.label_to_index = None
        self.smpc = smpc
        self.hash_index_map = {}  # To map hashes back to local indices
        self.embed_adata = self.embed_adata_file()
        del self.model

    def compute_local_distances(self, query_embeddings):
        """
        Computes distances between query embeddings and local reference embeddings.

        Supports both standard (non-SMPC) and SMPC-based modes.

        Args:
            query_embeddings (np.ndarray or crypten.CrypTensor):
                Query embeddings to compare against local reference data.
                Can be a plain NumPy array (non-SMPC) or a CrypTensor (SMPC).

        Returns:
            tuple:
                - distances (np.ndarray or crypten.CrypTensor): Top-k distances per query of shape (n_query, k).
                  Plain float array in non-SMPC mode; encrypted CrypTensor in SMPC mode.
                - hashed_indices (list of list of str): Hashed local reference indices for each query sample.
                  Used for anonymous voting.
        """
        if self.smpc:
            return self.compute_squared_distances(query_embeddings)
        # Perform local similarity search using faiss
        index = faiss.IndexFlatL2(self.embed_adata.obsm["X_scGPT"].shape[1])
        index.add(self.embed_adata.obsm["X_scGPT"])
        D, I = index.search(query_embeddings, self.k)
        hashed_indices = self.hash_indices(I)
        return D, hashed_indices

    def compute_squared_distances(self, secure_embeddings):
        """
        Computes encrypted squared distances and returns top-k per query.
        Also returns corresponding hashed reference indices.

        Returns:
            encrypted_topk (CrypTensor): shape (n_query, k)
            hashed_indices (list of list of str): shape (n_query, k)
        """
        # distances = []
        reference = torch.tensor(
            self.embed_adata.obsm["X_scGPT"],
            dtype=torch.float32,
            device=self.device,
        )
        reference = crypten.cryptensor(reference)
        query_norm = secure_embeddings.square().sum(dim=1).unsqueeze(1)  # (n_query, 1)
        ref_norm = reference.square().sum(dim=1).unsqueeze(0)  # (1, n_ref)

        # Step 2: Compute dot product
        cross = secure_embeddings @ reference.transpose(0, 1)  # (n_query, n_ref)

        # Step 3: Compute pairwise distances: ||x - y||² = ||x||² + ||y||² - 2x·y
        distances = query_norm + ref_norm - 2 * cross
        del reference, query_norm, ref_norm, cross
        encrypted_topk, topk_indices = top_k_encrypted_distances(distances, self.k)
        topk_mapped_ind = [(k @ self.enc_celltype_ind_offset.unsqueeze(1)).squeeze(1) for k in topk_indices]
        topk_mapped_ind = crypten.stack(topk_mapped_ind, dim=1)
        return encrypted_topk, topk_mapped_ind

    def hash_indices(self, indices):
        """
        Hashes reference indices with a client-specific salt to ensure
        privacy-preserving, non-linkable identifiers. This prevents
        the server or other clients from inferring or correlating local
        data structure, even if raw index values are similar across sites.

        Args:
            indices (np.ndarray): The indices to hash.

        Returns:
            list: A list of hashes corresponding to each set of indices.
        """
        client_salt = self.get_client_salt()  # Assume there's a method to get a client-specific salt
        hash_I = []

        for query_set in indices:
            hash_list = []
            for index in query_set:
                # Include client-specific salt in the hash
                index_hash = hashlib.sha256(f"{client_salt}-{index}".encode()).hexdigest()
                hash_list.append(index_hash)
                self.hash_index_map[index_hash] = index  # Store the map locally
            hash_I.append(hash_list)

        return hash_I


    def get_client_salt(self):
        """
        Returns a unique client-specific salt for hashing.

        This could be based on a client ID, a unique identifier, or
        some other property that makes the client distinct.

        Returns:
            str: A unique salt for the client.
        """
        return str(self.log_id)

    # def vote(self, global_nearest_samples):
    #     """
    #    Perform per-query voting based on the client's local reference data.
    #
    #     For each query cell, votes are cast based on the labels of locally matched
    #     reference cells whose hashed indices appear in the global nearest neighbor list.
    #
    #     Args:
    #         global_nearest_samples (List[List[str]]):
    #             A list where each element corresponds to a query cell and contains
    #             the hashed indices of its top-k nearest neighbors (combined across clients).
    #
    #     Returns:
    #         Union[List[Dict[str, int]], List[crypten.CrypTensor]]:
    #             - If SMPC is disabled: returns a list of dictionaries (one per query) where each key is a label and
    #               the value is the count of votes.
    #             - If SMPC is enabled: returns a list of encrypted fixed-length vote vectors (CrypTensors),
    #               where each index corresponds to a global label index.
    #     """
    #     votes = []
    #     if self.smpc:
    #         n_queries = global_nearest_samples.size(0)
    #         local_ind = self.enc_celltype_ind_offset.unsqueeze(0)
    #         # ct_labels_enc = crypten.cryptensor(torch.tensor(self.mapped_ct, dtype=torch.float32, device=self.device))
    #         ct_labels_enc = crypten.cryptensor(torch.tensor(self.mapped_ct, dtype=torch.long, device=self.device))
    #         ct_labels_exp = ct_labels_enc.unsqueeze(0).expand(n_queries, self.n_samples)
    #         for k in range(self.k):
    #             sample_k = global_nearest_samples[:, k].unsqueeze(1).expand(n_queries, self.n_samples)
    #             match_mask = (sample_k == local_ind)
    #             votes.append((match_mask * ct_labels_exp).sum(dim=1).unsqueeze(1))
    #         votes = crypten.cat(votes, dim=1)
    #     else:
    #         for query_sample in global_nearest_samples:
    #             vote_counts = {}
    #             for hash_value in query_sample:
    #                 if hash_value in self.hash_index_map:
    #                     local_index = self.hash_index_map[hash_value]
    #                     label = self.adata.obs[self.celltype_key].values[local_index]
    #                     if label not in vote_counts:
    #                         vote_counts[label] = 0
    #                     vote_counts[label] += 1
    #             votes.append(vote_counts)
    #     return votes
    def vote(self, global_nearest_samples):
        n_queries = global_nearest_samples.size(0)
        n_classes = len(self.label_to_index)
        ct_label_tensor = torch.tensor(self.mapped_ct, dtype=torch.long, device=self.device)
        ct_onehot = torch.nn.functional.one_hot(ct_label_tensor, num_classes=n_classes)
        ct_onehot_enc = crypten.cryptensor(ct_onehot)
        votes = []
        for k_idx in range(self.k):
            mask = global_nearest_samples[:, k_idx].unsqueeze(2)
            votes.append((mask * ct_onehot_enc.unsqueeze(0)).sum(dim=1))
        return sum(votes)

    def report_celltypes(self):
        """
        Report the unique cell types present in this client's reference data.

        Returns:
            set of str: Unique cell type labels.
        """
        labels = self.adata.obs[self.celltype_key].unique()
        return set(labels)

    def harmonize_celltypes(self, global_label_to_index, ind_offset):
        """
        Maps local cell types to global label indices with an offset and encrypts them.

        Args:
            global_label_to_index (dict): Maps each cell type string to a unique integer index.
            ind_offset (int): Offset to make local labels globally unique.
        """
        # TODO: Check its effect on fedetated without SMPC
        self.label_to_index = global_label_to_index
        cell_types = self.embed_adata.obs[self.celltype_key].values
        self.mapped_ct = [global_label_to_index[ct] for ct in cell_types]
        self.ind_offset = ind_offset
        # global_indices =  np.arange(self.n_samples)+ ind_offset
        # self.celltype_ind_offset = global_indices
        # self.enc_celltype_ind_offset = crypten.cryptensor(torch.tensor(global_indices, dtype=torch.float32, device=self.device))
        if self.smpc:
            global_indices = torch.arange(self.n_samples, dtype=torch.long, device=self.device) + ind_offset
            self.enc_celltype_ind_offset = crypten.cryptensor(global_indices)


    def report_n_local_samples(self):
        return crypten.cryptensor(torch.tensor(self.n_samples, dtype=torch.float32, device=self.device))


class FedEmbedder(FedBase):
    def __init__(self, data_dir, reference_adata, query_adata, output_dir, k, smpc=False, **kwargs):
        super().__init__(data_dir=data_dir, output_dir=output_dir, **kwargs)
        self.label_to_index = None
        self.index_to_label = None
        self.smpc = smpc
        self.k = k
        adata = read_h5ad(data_dir, reference_adata)
        self.distribute_adata_by_batch(adata, kwargs['batch_key'], keep_vars=True)
        self.celltype_key = kwargs['celltype_key']
        for c in range(self.n_clients):
            client = ClientEmbedder(reference_adata='adata.h5ad',
                                    data_dir=self.clients_data_dir[c],
                                    output_dir=self.clients_output_dir[c],
                                    log_id=f"client_{self.client_ids[c]}",
                                    k=self.k,
                                    logger=self.logger,
                                    smpc=smpc,
                                    **kwargs)
            self.clients.append(client)
        self.query, self.embed_query = self.embed_query_adata(query_adata, output_dir=output_dir, data_dir=data_dir, **kwargs)
        if self.smpc:
            embed_query = torch.tensor(self.embed_query.obsm["X_scGPT"], dtype=torch.float32, device=self.device)
            self.embed_query.obsm['secure_embed'] = crypten.cryptensor(embed_query)


    def embed_query_adata(self, query_adata, **kwargs):
        embedder = Embedder(reference_adata=query_adata, k=self.k, **kwargs)
        embedder.embed_adata_file()
        del embedder.model
        return embedder.adata, embedder.embed_adata


    def global_aggregate_distances(self, client_distances, client_indices):
        """
        Aggregates distance information from all clients and determines the global top-k nearest neighbors
        for each query sample.

        Supports both standard (non-SMPC) and SMPC-based modes.

        Args:
            client_distances (list):
                - If SMPC is disabled: list of np.ndarray, each of shape (n_query, k)
                - If SMPC is enabled: list of CrypTensor, each of shape (n_query, k)
            client_indices (list of list of list of str):
                Hashed reference indices from each client. Shape: (n_clients, n_query, k)

        Returns:
            list of list of str:
                Global top-k hashed reference indices per query sample. Shape: (n_query, k)
        """
        if self.smpc:
            return self.secure_top_k_distance_agg(client_distances, client_indices)
        all_distances = np.hstack(client_distances)
        all_hashes = np.hstack(client_indices)
        sorted_indices = np.argsort(all_distances, axis=1)
        k_nearest_samples = []
        for r, row in enumerate(sorted_indices):
            temp = []
            for ind in row[:self.k]:
                temp.append(all_hashes[r][ind])
            k_nearest_samples.append(temp)
        return k_nearest_samples

    def secure_top_k_distance_agg(self, client_distances, client_indices):
        """
        Securely aggregates encrypted top-k distances from all clients and returns
        global top-k hashed reference indices per query cell.

        Args:
            client_distances (list of CrypTensor): Encrypted local top-k distance matrices from clients.
            client_hashes (list of list of list of str): Corresponding hashed reference indices.

        Returns:
            list[list[str]]: Global top-k hashed reference IDs for each query cell.
        """
        distances = crypten.cat(client_distances, dim=1)
        indices = crypten.cat(client_indices, dim=1)
        one_hot_indices = top_k_ind_selection(distances.clone(), self.k)
        top_k_indices = [(one_hot_indices[k] * indices).sum(dim=1).unsqueeze(dim=1) for k in range(self.k)]
        k_nearest_samples = crypten.cat(top_k_indices, dim=1)
        return k_nearest_samples

    # def aggregate_client_votes(self, client_votes):
    #     """
    #     Aggregate the vote counts from all clients.
    #
    #     Args:
    #         client_votes (list of dict): A list of dictionaries containing vote counts from each client.
    #
    #     Returns:
    #         np.ndarray: The final predicted labels for the query data.
    #     """
    #     if self.smpc:
    #         import pdb; pdb.set_trace()
    #         aggregated_votes = crypten.stack(client_votes, dim=2).sum(dim=2)
    #         pred_labels, _ = aggregated_votes.max(dim=1)
    #         pred_labels_plain = pred_labels.get_plain_text().cpu().numpy().astype('int')
    #         pred_labels_plain = np.array([self.index_to_label[label] for label in pred_labels_plain], dtype=object)
    #         return pred_labels_plain
    #
    #     aggregated_votes = [{} for _ in range(self.embed_query.shape[0])]
    #     for client_vote in client_votes:
    #         for r, sample in enumerate(client_vote):
    #             for label, count in sample.items():
    #                 temp = aggregated_votes[r]
    #                 if label not in temp:
    #                     temp[label] = 0
    #                 temp[label] += count
    #                 aggregated_votes[r] = temp
    #
    #     # Determine the label with the most votes for each query point
    #     final_predictions = []
    #     for r in range(self.embed_query.shape[0]):
    #         if aggregated_votes[r]:
    #             final_label = max(aggregated_votes[r], key=aggregated_votes[r].get)
    #             final_predictions.append(final_label)
    #         else:
    #             final_predictions.append(None)  # Handle cases where there are no votes
    #
    #     return np.array(final_predictions)

    def aggregate_client_votes(self, client_votes):
        if self.smpc:
            total = crypten.stack(client_votes, dim=2).sum(dim=2)
            pred, _ = total.max(dim=1)
            labels = pred.get_plain_text().cpu().numpy().astype(int)
            return np.array([self.index_to_label[i] for i in labels], dtype=object)
        aggregated = [{} for _ in range(self.embed_query.shape[0])]
        for votes in client_votes:
            for i, vc in enumerate(votes):
                for label, count in vc.items():
                    aggregated[i][label] = aggregated[i].get(label, 0) + count
        final = []
        for vc in aggregated:
            final.append(max(vc, key=vc.get) if vc else None)
        return np.array(final, dtype=object)

    def federated_reference_map(self):
        """
        Perform the federated reference mapping process.

        Returns:
            tuple: Ground truth labels, predicted labels.
        """
        self.collect_global_celltypes()
        client_distances, client_indices = [], []
        query_embedding = self.embed_query.obsm["secure_embed" if self.smpc else "X_scGPT"]
        for client in self.clients:
            distances, indices = client.compute_local_distances(query_embedding)
            client_distances.append(distances)
            client_indices.append(indices)

        # Aggregate distances and perform majority voting
        k_nearest_samples = self.global_aggregate_distances(client_distances, client_indices)
        client_votes = []

        # Each client computes votes based on the query embeddings
        for client in self.clients:
            votes = client.vote(k_nearest_samples)
            client_votes.append(votes)
        preds = self.aggregate_client_votes(client_votes)
        gt = self.query.obs[self.celltype_key].to_numpy()
        return gt, preds

    def collect_global_celltypes(self):
        """
        Collects all unique cell types from clients and builds a consistent global label index.

        Sets:
            self.label_to_index (dict): Mapping from cell type label (or hash in SMPC) to global index.
            self.index_to_label (dict): Reverse mapping for predictions.
        """
        all_labels = set()
        for client in self.clients:
            all_labels |= client.report_celltypes()

        # Build mappings (sorted for deterministic ordering)
        sorted_labels = sorted(list(all_labels))
        self.label_to_index = {label: idx for idx, label in enumerate(sorted_labels, 1)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        if self.smpc:
            self.aggregate_total_n_samples()
            client_offset = 0
            for client in self.clients:
                client.harmonize_celltypes(self.label_to_index, client_offset)
                client_offset += self.total_n_samples
        else:
            for client in self.clients:
                client.harmonize_celltypes(self.label_to_index, 0)


    def aggregate_total_n_samples(self):
        encrypted_counts = [
            client.report_n_local_samples()
            for client in self.clients
        ]
        total = encrypted_counts[0]
        for count in encrypted_counts[1:]:
            total = total + count

        self.total_n_samples = int(total.get_plain_text().item())
