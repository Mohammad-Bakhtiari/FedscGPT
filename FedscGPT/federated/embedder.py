import scgpt as scg
import faiss
import hashlib
import numpy as np
from FedscGPT.base import FedBase, BaseMixin
from FedscGPT.utils import read_h5ad
from FedscGPT.centralized.embedder import Embedder


class ClientEmbedder(Embedder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hash_index_map = {}  # To map hashes back to local indices
        self.embed_adata = self.embed_adata_file()

    def compute_local_distances(self, query_embeddings):
        """
        Computes local distances between the local reference embeddings and the query embeddings.

        Args:
            query_embeddings (np.ndarray): The query embeddings.
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            tuple: Distances and hashed indices of the nearest neighbors.
        """
        # Perform local similarity search using faiss
        index = faiss.IndexFlatL2(self.embed_adata.obsm["X_scGPT"].shape[1])
        index.add(self.embed_adata.obsm["X_scGPT"])
        D, I = index.search(query_embeddings, self.k)
        # Hash the indices
        hashed_indices = self.hash_indices(I)

        return D, hashed_indices

    def hash_indices(self, indices):
        """
        Hashes the local indices for communication with the server,
        including a client-specific salt to ensure uniqueness.

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
        Perform a majority vote on the local embeddings based on the nearest neighbors.

        Args:
            hashed_indices (list): The hashed indices of the nearest neighbors.

        Returns:
            dict: A dictionary with labels as keys and vote counts as values.
        """
        vote = []
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


class FedEmbedder(FedBase):
    def __init__(self, data_dir, reference_adata, query_adata, output_dir, k, **kwargs):
        super().__init__(data_dir=data_dir, output_dir=output_dir, **kwargs)
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
                                    logger=self.logger, **kwargs)
            self.clients.append(client)
        self.query, self.embed_query = self.embed_query_adata(query_adata, output_dir=output_dir, data_dir=data_dir, **kwargs)


    def embed_query_adata(self, query_adata, **kwargs):
        embedder = Embedder(reference_adata=query_adata, k=self.k, **kwargs)
        embedder.embed_adata_file()
        return embedder.adata, embedder.embed_adata


    def global_aggregate_distances(self, client_distances, client_hashes):
        """
        Aggregate distances from all clients and determine the global nearest neighbors.

        Args:
            client_distances (list of np.ndarray): List of distances from each client.
            client_hashes (list of list): List of hashed indices from each client.

        Returns:
            tuple: Global distances and aggregated hashes of the nearest neighbors.
        """
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


    def aggregate_client_votes(self, client_votes):
        """
        Aggregate the vote counts from all clients.

        Args:
            client_votes (list of dict): A list of dictionaries containing vote counts from each client.

        Returns:
            np.ndarray: The final predicted labels for the query data.
        """
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
        client_distances, client_hashes = [], []

        # Each client computes distances to its local reference embeddings
        for client in self.clients:
            distances, hashed_indices = client.compute_local_distances(self.embed_query.obsm["X_scGPT"])
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