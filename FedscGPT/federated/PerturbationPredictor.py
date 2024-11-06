import os
from copy import deepcopy
import numpy as np
from gears import PertData
from sympy.physics.units import percent

from FedscGPT.centralized.PerturbationPredictor import Training
from FedscGPT.base import FedBase, BaseClientMixin
from FedscGPT.federated.aggregator import FedAvg
from FedscGPT.utils import SEED, get_logger

class ClientPerturbationPredictor(Training, BaseClientMixin):
    def __init__(self, **kwargs):
        Training.__init__(self, **kwargs)
        self.n_samples = len(self.adata)




class FedPerturbationPredictor(FedBase):
    def __init__(self, n_clients, data_dir, output_dir, **kwargs):
        super().__init__(data_dir=data_dir, output_dir=output_dir, **kwargs)
        self.n_clients = n_clients
        self.clients_data_dir = os.path.join(data_dir, f"{n_clients}_clients")
        self.setup_clients(**kwargs)

    def setup_clients(self, reference_adata, **kwargs):
        self.client_ids = list(range(self.n_clients)) if self.client_ids is None else self.client_ids
        self.logger = get_logger(self.output_dir, "FedscGPT", self.client_ids)
        clients_data_dir = [
            item for item in os.listdir(self.clients_data_dir)
            if os.path.isdir(os.path.join(self.clients_data_dir, item))
        ]
        assert len(clients_data_dir) == self.n_clients, f"Number of clients directories {len(clients_data_dir)} does not match the number of clients {self.n_clients}"
        self.clients_data_dir = [os.path.join(self.clients_data_dir, d) for d in clients_data_dir]
        self.clients_output_dir = self.create_dirs(self.output_dir)
        if os.path.isabs(reference_adata):
            reference_adata = os.path.basename(reference_adata)
        for c in range(self.n_clients):
            client = ClientPerturbationPredictor(reference_adata=reference_adata,
                                                 data_dir=self.clients_data_dir[c],
                                                 output_dir=self.clients_output_dir[c],
                                                 log_id=f"client_{self.client_ids[c]}",
                                                 logger=self.logger, **kwargs)
            self.clients.append(client)
        self.retain_best_model_retain(False)



class FedPerturbationPredictorTraining(FedPerturbationPredictor, FedAvg):
    def __init__(self, **kwargs):
        FedPerturbationPredictor.__init__(self, **kwargs)
        FedAvg.__init__(self, self.fed_config.n_rounds)
        for client in self.clients:
            client.setup()
            client.setup_losses()
            client.move_model_to_cpu()