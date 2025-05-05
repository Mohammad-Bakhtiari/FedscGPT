from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
import torch


class Aggregator(ABC):
    """
    Abstract class for implementing aggregation logic in federated learning or
    other distributed systems where model parameters need to be aggregated.
    """

    def __init__(self, n_rounds, smpc=False, debug=False, **kwargs):
        """
        Initializes the aggregator with the specified number of rounds.

        Parameters:
        - n_rounds (int): Maximum number of aggregation rounds.
        """
        print(f"Aggregator initialized with n_rounds={n_rounds}, smpc={smpc}, debug={debug}")
        self.n_rounds = n_rounds
        self.current_round = 0
        self.smpc = smpc
        self.debug = debug

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

    def get_global_decrypted(self, encrypted_weights: List) -> Dict[str, List[torch.tensor]]:
        """
        Decrypts and updates global weights from a list of encrypted tensors.

        Args:
            encrypted_weights (List): List of encrypted tensors corresponding to model parameters.
        """
        decrypted_weights = {
            key: encrypted_weights[i].get_plain_text().clone().detach().to(torch.float32)
            for i, key in enumerate(self.global_model_keys)
        }
        return decrypted_weights


    def update_global_weights(self, weights: Dict[str, List[torch.Tensor]]) -> None:
        if self.debug:
            assert self.global_model_keys == list(weights.keys()), f"Key mismatch: {self.global_model_keys} vs {list(weights.keys())}"
            for key in self.global_model_keys:
                assert self.global_weight_shapes[key] == weights[key].shape, f"Shape mismatch for {key}: {self.global_weight_shapes[key]} vs {weights[key].shape}"
        assert self.global_model_keys is not None, "Global model keys are not set. Call set_global_weight_struct() first."
        self.global_weights = weights

class FedAvg(Aggregator):
    def __init__(self, weighted, **kwargs):
        """
        Initializes the FedAvg aggregator with the specified number of rounds.

        """
        print(f"FedAvg aggregator initialized with weighted={weighted}")
        super().__init__(**kwargs)
        self.weighted = weighted

    def aggregate(self, local_weights, **kwargs):
        """
        Dispatch to the appropriate aggregation method.
        """
        if self.smpc:
            self.aggregate_smpc(local_weights)
        else:
            n_local_samples = kwargs.get("n_local_samples", None)
            self.aggregate_plain(local_weights, n_local_samples)

    def aggregate_plain(self, local_weights: List[Dict[str, torch.Tensor]],
                        n_local_samples: Optional[List[int]] = None) -> None:
        """
        Aggregation without SMPC.

        Args:
            local_weights (List[Dict[str, torch.Tensor]]): List of state_dicts from clients.
            n_local_samples (List[int], optional): Number of samples per client for weighting.
        """
        n_clients = len(local_weights)

        if self.weighted:
            assert n_local_samples is not None, "Missing 'n_local_samples' for weighted aggregation"
            sample_ratios = [n / sum(n_local_samples) for n in n_local_samples]

        global_weights = {}
        for param in local_weights[0].keys():
            if self.weighted:
                global_weights[param] = torch.stack(
                    [local_weights[i][param] * sample_ratios[i] for i in range(n_clients)]
                ).sum(0)
            else:
                global_weights[param] = torch.stack(
                    [local_weights[i][param] for i in range(n_clients)]
                ).sum(0) / n_clients
        self.update_global_weights(global_weights)

    def aggregate_smpc(self, encrypted_weights_list: List[List]) -> None:
        """
        Aggregation with SMPC.

        Args:
            encrypted_weights_list (List[List]): Each client's list of encrypted tensors.

        - If weighted=True: sample ratios already applied on client side.
        - If weighted=False: perform vanilla averaging server-side.
        """
        n_clients = len(encrypted_weights_list)
        summed = [sum(param_group) for param_group in zip(*encrypted_weights_list)]

        if self.weighted:
            # Weights are already scaled client-side
            global_weights = self.get_global_decrypted(summed)
        else:
            averaged = [param / n_clients for param in summed]
            global_weights = self.get_global_decrypted(averaged)
        self.update_global_weights(global_weights)

    def stop(self, **kwargs):
        """
        Stop if the maximum number of rounds has been reached.

        Returns:
        - bool: True if the stopping condition is met, otherwise False.
        """
        self.current_round += 1
        return self.current_round >= self.n_rounds
