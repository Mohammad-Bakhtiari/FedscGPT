import scgpt as scg
import faiss
import hashlib
import numpy as np
import torch
import crypten
from FedscGPT.base import FedBase, BaseMixin
from FedscGPT.utils import (read_h5ad, smpc_encrypt_embedding, concat_encrypted_distances, top_k_encrypted_distances,
                            get_plain_indices, top_k_ind_selection, encrypted_present_hashes)
from FedscGPT.centralized.embedder import Embedder


class ClientEmbedder(Embedder):
    def __init__(self, smpc=False, **kwargs):
        super().__init__(**kwargs)
        self.label_to_index = None
        self.smpc = smpc
        self.hash_index_map = {}  # To map hashes back to local indices
        self.embed_adata = self.embed_adata_file()

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
        distances = []
        reference = torch.tensor(
            self.embed_adata.obsm["X_scGPT"],
            dtype=torch.float32,
            device=self.device
        )
        reference = crypten.cryptensor(reference)
        for ref_vector in reference:
            diff = secure_embeddings - ref_vector
            sq_diff = diff * diff
            d = sq_diff.sum(dim=1)  # (n_query,)
            distances.append(d.unsqueeze(1))

        encrypted_dist_matrix = concat_encrypted_distances(distances)
        encrypted_topk, topk_indices = top_k_encrypted_distances(encrypted_dist_matrix, self.k)
        hashed_indices = self.hash_indices(get_plain_indices(topk_indices))
        return encrypted_topk, hashed_indices

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

    def vote(self, global_nearest_samples):
        """
       Perform per-query voting based on the client's local reference data.

        For each query cell, votes are cast based on the labels of locally matched
        reference cells whose hashed indices appear in the global nearest neighbor list.

        Args:
            global_nearest_samples (List[List[str]]):
                A list where each element corresponds to a query cell and contains
                the hashed indices of its top-k nearest neighbors (combined across clients).

        Returns:
            Union[List[Dict[str, int]], List[crypten.CrypTensor]]:
                - If SMPC is disabled: returns a list of dictionaries (one per query) where each key is a label and
                  the value is the count of votes.
                - If SMPC is enabled: returns a list of encrypted fixed-length vote vectors (CrypTensors),
                  where each index corresponds to a global label index.
        """
        vote = []
        if self.smpc:
            n_labels = len(self.label_to_index)
            for query_sample in global_nearest_samples:
                vote_vector = torch.zeros(n_labels)
                for hash_value in query_sample:
                    if hash_value in self.hash_index_map:
                        local_index = self.hash_index_map[hash_value]
                        label = self.adata.obs[self.celltype_key].values[local_index]
                        label_idx = self.label_to_index[label]  # label could be hash
                        vote_vector[label_idx] += 1
                vote.append(crypten.cryptensor(vote_vector))
        else:
            for query_sample in global_nearest_samples:
                vote_counts = {}
                for hash_value in query_sample:
                    if hash_value in self.hash_index_map:
                        local_index = self.hash_index_map[hash_value]
                        label = self.adata.obs[self.celltype_key].values[local_index]
                        if label not in vote_counts:
                            vote_counts[label] = 0
                        vote_counts[label] += 1
                vote.append(vote_counts)
        return vote

    def report_celltypes(self, global_hash_to_index=None):
        """
        Report the unique cell types present in this client's reference data.

        Returns:
            set of str: Unique cell type labels.
        """
        labels = self.adata.obs[self.celltype_key].unique()
        if self.smpc:
            assert global_hash_to_index is not None, "Global hash to index mapping is required for SMPC mode."
            return encrypted_present_hashes(global_hash_to_index, labels)
        return set(labels)

    def hash_labels(self):
        return [hashlib.sha256(label.encode()).hexdigest() for label in set(self.adata.obs[self.celltype_key].unique())]



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
            smpc_encrypt_embedding(self.embed_query)


    def embed_query_adata(self, query_adata, **kwargs):
        embedder = Embedder(reference_adata=query_adata, k=self.k, **kwargs)
        embedder.embed_adata_file()
        return embedder.adata, embedder.embed_adata


    def global_aggregate_distances(self, client_distances, client_hashes):
        """
        Aggregates distance information from all clients and determines the global top-k nearest neighbors
        for each query sample.

        Supports both standard (non-SMPC) and SMPC-based modes.

        Args:
            client_distances (list):
                - If SMPC is disabled: list of np.ndarray, each of shape (n_query, k)
                - If SMPC is enabled: list of CrypTensor, each of shape (n_query, k)
            client_hashes (list of list of list of str):
                Hashed reference indices from each client. Shape: (n_clients, n_query, k)

        Returns:
            list of list of str:
                Global top-k hashed reference indices per query sample. Shape: (n_query, k)
        """
        if self.smpc:
            return self.secure_top_k_distance_agg(client_distances, client_hashes)
        all_distances = np.hstack(client_distances)
        all_hashes = np.hstack(client_hashes)
        sorted_indices = np.argsort(all_distances, axis=1)
        k_nearest_samples = []
        for r, row in enumerate(sorted_indices):
            temp = []
            for ind in row[:self.k]:
                temp.append(all_hashes[r][ind])
            k_nearest_samples.append(temp)
        return k_nearest_samples

    def secure_top_k_distance_agg(self, client_distances, client_hashes):
        """
        Securely aggregates encrypted top-k distances from all clients and returns
        global top-k hashed reference indices per query cell.

        Args:
            client_distances (list of CrypTensor): Encrypted local top-k distance matrices from clients.
            client_hashes (list of list of list of str): Corresponding hashed reference indices.

        Returns:
            list[list[str]]: Global top-k hashed reference IDs for each query cell.
        """
        # Combine all (n_query, k) CrypTensors → shape (n_query, k * n_clients)
        encrypted_concat = concat_encrypted_distances(client_distances)
        all_hashes = sum(client_hashes, [])  # flatten across clients → list of (n_query, k)
        # Secure top-k from encrypted distances
        topk_indices = top_k_ind_selection(encrypted_concat.clone(), self.k, encrypted_concat.size(1))

        topk_indices = get_plain_indices(topk_indices)
        k_nearest_samples = []
        for i, row in enumerate(all_hashes):
            row_hashes = []
            for ind in topk_indices[i]:
                row_hashes.append(row[ind])
            k_nearest_samples.append(row_hashes)
        return k_nearest_samples

    def aggregate_client_votes(self, client_votes):
        """
        Aggregate the vote counts from all clients.

        Args:
            client_votes (list of dict): A list of dictionaries containing vote counts from each client.

        Returns:
            np.ndarray: The final predicted labels for the query data.
        """
        if self.smpc:
            n_queries = len(client_votes[0])
            n_labels = len(self.label_to_index)
            # Initialize encrypted vote matrix
            encrypted_vote_matrix = [crypten.cryptensor(torch.zeros(n_labels)) for _ in range(n_queries)]

            # Add votes securely across clients
            for client_vote in client_votes:
                for i in range(n_queries):
                    encrypted_vote_matrix[i] += client_vote[i]

            # Decrypt and determine predicted label
            final_predictions = []
            for enc_votes in encrypted_vote_matrix:
                plain_votes = enc_votes.get_plain_text().tolist()
                max_idx = int(np.argmax(plain_votes))
                final_predictions.append(self.index_to_label[max_idx])  # label may be hash or raw

            return np.array(final_predictions)

        aggregated_votes = [{} for _ in range(self.embed_query.shape[0])]
        for client_vote in client_votes:
            for r, sample in enumerate(client_vote):
                for label, count in sample.items():
                    temp = aggregated_votes[r]
                    if label not in temp:
                        temp[label] = 0
                    temp[label] += count
                    aggregated_votes[r] = temp

        # Determine the label with the most votes for each query point
        final_predictions = []
        for r in range(self.embed_query.shape[0]):
            if aggregated_votes[r]:
                final_label = max(aggregated_votes[r], key=aggregated_votes[r].get)
                final_predictions.append(final_label)
            else:
                final_predictions.append(None)  # Handle cases where there are no votes

        return np.array(final_predictions)

    def federated_reference_map(self):
        """
        Perform the federated reference mapping process.

        Returns:
            tuple: Ground truth labels, predicted labels.
        """
        self.collect_global_celltypes()
        client_distances, client_hashes = [], []
        query_embedding = self.embed_query.obsm["secure_embed" if self.smpc else "X_scGPT"]
        for client in self.clients:
            distances, hashed_indices = client.compute_local_distances(query_embedding)
            client_distances.append(distances)
            client_hashes.append(hashed_indices)

        # Aggregate distances and perform majority voting
        k_nearest_samples = self.global_aggregate_distances(client_distances, client_hashes)
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
        if self.smpc:
            # Step 1: Build global hash index
            self.global_hash_to_index()

            # Step 2: Secure aggregation of presence
            encrypted_presence = self.clients[0].report_celltypes(self.hash_to_index)
            for client in self.clients[1:]:
                encrypted_presence += client.report_celltypes(self.hash_to_index)

            global_presence = encrypted_presence.get_plain_text().int()

            # Step 3: Build final label set based on presence
            all_labels = []
            for h, i in self.hash_to_index.items():
                if global_presence[i] > 0:
                    all_labels.append(h)

        else:
            # Non-SMPC: Collect plain labels
            all_labels = set()
            for client in self.clients:
                all_labels |= client.report_celltypes()

        # Build mappings (sorted for deterministic ordering)
        sorted_labels = sorted(list(all_labels))
        self.label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

        # Broadcast to all clients
        for client in self.clients:
            client.label_to_index = self.label_to_index

    def global_hash_to_index(self):
        hashed_labels = set()
        for client in self.clients:
            hashed_labels |= set(client.hash_labels())  # returns set of hashes
        hashed_labels = sorted(list(hashed_labels))
        self.hash_to_index = {h: i for i, h in enumerate(hashed_labels)}