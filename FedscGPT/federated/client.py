from FedscGPT.utils import check_weights_nan
import crypten
import torch

class Client:
    def __init__(self, n_total_samples, smpc=False, debug=False, weighted=False, **kwargs):
        self.smpc = smpc
        self.n_samples = self.adata.X.shape[0]
        self.sample_ration = self.n_samples / n_total_samples
        self.debug = debug
        self.weighted = weighted

    def get_local_updates(self):
        """Get the local updates.

        Returns
        -------
        dict, int : Without SMPC
        list of crypten.cryptensor, crypten.cryptensor : With SMPC
        """
        weights = self.get_weights()
        if self.smpc:
            weights = weights.values()
            if self.weighted:
               weights = [param * self.sample_ration for param in weights]
            check_weights_nan(weights, "after training", self.debug)
            encrypted_weights = [crypten.cryptensor(param) for param in weights]
            return encrypted_weights
        return weights

    def get_weights(self):
        """ Get the weights of the model
        Returns
        -------
        dict
            The weights of the model
        """
        return self.model.state_dict()

    def set_weights(self, state_dict):
        """ Set the weights of the model
        Parameters
        ----------
        state_dict: dict
            The weights of the model
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state_dict:
                    param.data.copy_(state_dict[name])
