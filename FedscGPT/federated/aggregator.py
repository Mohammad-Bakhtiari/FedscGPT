from FedscGPT import utils
utils.set_seed()
from abc import ABC, abstractmethod
import torch


class Aggregator(ABC):
    """
    Abstract class for implementing aggregation logic in federated learning or
    other distributed systems where model parameters need to be aggregated.
    """

    def __init__(self, n_rounds):
        """
        Initializes the aggregator with the specified number of rounds.

        Parameters:
        - n_rounds (int): Maximum number of aggregation rounds.
        """
        self.n_rounds = n_rounds
        self.current_round = 0
        self.global_weights = {}

    @abstractmethod
    def aggregate(self, local_weights, **kwargs):
        """
        Abstract method to aggregate local model weights.

        Parameters:
        - local_weights (list): A list of model weights from all participating clients.

        Must be implemented by subclasses to specify how local weights are combined.
        """
        pass

    @abstractmethod
    def stop(self, **kwargs):
        """
        Abstract method to determine if the aggregation process should stop.

        Returns:
        - (bool): True if the stopping condition is met, False otherwise.

        Must be implemented by subclasses to specify the stopping criteria.
        """
        pass


class FedAvg(Aggregator):
    def aggregate(self, local_weights, **kwargs):
        n_clients = len(local_weights)
        self.global_weights = {}
        for param in local_weights[0].keys():
            self.global_weights[param] = torch.stack(
                [client[param] for client in local_weights]).sum(0) / n_clients

    def weighted_aggregate(self, local_weights, n_local_samples, **kwargs):
        """
        Aggregate local weights by computing a weighted average based on the number of samples each client has.

        Parameters:
        - local_weights (list of list of numpy.ndarray): The list of model weights from each client.
        - n_local_samples (list of int): The list of number of data samples from each client.

        Returns:
        - numpy.ndarray: The aggregated global weights.
        """
        sample_ratios = [n / sum(n_local_samples) for n in n_local_samples]
        for param in local_weights[0].keys():
            self.global_weights[param] = torch.stack(
                [local_weights[i][param] * sample_ratios[i] for i in range(len(local_weights))]).sum(0)


    def stop(self, **kwargs):
        """
        Stop if the maximum number of rounds has been reached.

        Returns:
        - bool: True if the stopping condition is met, otherwise False.
        """
        self.current_round += 1
        return self.current_round >= self.n_rounds
