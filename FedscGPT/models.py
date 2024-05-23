import torch.nn as nn
import torch
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.model import TransformerModel, AdversarialDiscriminator


class BaseMixin:
    def __init__(self, lr, schedule_interval, schedule_ratio, amp):
        self.lr = lr
        self.optimizers = {}
        self.losses = {}
        self.lr_schedulers = {}
        self.schedule_interval = schedule_interval
        self.schedule_ratio = schedule_ratio
        self.amp = amp
        self.losses = {'default': masked_mse_loss,
                       'cls': nn.CrossEntropyLoss(),
                       'dab': nn.CrossEntropyLoss()
                       }
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.set_defualt()

    def set_DAB_sep_optim(self):
        self.optimizers['dab'] = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_schedulers['dab'] = torch.optim.lr_scheduler.StepLR(
            self.optimizers['dab'], self.schedule_interval, gamma=self.schedule_ratio
        )

    def set_defualt(self):
        self.optimizers['optimizer'] = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, eps=1e-4 if self.amp else 1e-8
        )
        self.lr_schedulers['scheduler'] = torch.optim.lr_scheduler.StepLR(
            self.optimizers['optimizer'], self.schedule_interval, gamma=self.schedule_ratio
        )

    def set_ADV(self):
        self.losses["adv"] = nn.CrossEntropyLoss()  # consider using label smoothing
        self.optimizers["E"] = torch.optim.Adam(self.model.parameters(), lr=self.lr_ADV)
        self.lr_schedulers["E"] = torch.optim.lr_scheduler.StepLR(
            self.optimizers['E'], self.schedule_interval, gamma=self.schedule_ratio
        )
        self.optimizers["D"] = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_ADV)
        self.lr_schedulers["D"] = torch.optim.lr_scheduler.StepLR(
            self.optimizers['D'], self.schedule_interval, gamma=self.schedule_ratio
        )

    def configure_optimizers(self, DAB=False, ADV=False, embsize=None, n_cls=None):
        """
        Configures optimizers and losses for training
        Order:
        1. Adversarial discriminator (if DAB) - separate optimizer
        2. DAB - separate optimizer
        Parameters
        ----------
        DAB
        ADV
        embsize
        n_cls

        Returns
        -------

        """
        if ADV:
            if embsize is None or n_cls is None:
                raise ValueError("embsize and n_cls must be provided for adversarial training")
            self.build_adv_discriminator(embsize, n_cls)
            self.set_ADV()

        if DAB:
            self.set_DAB_sep_optim()

    def build_adv_discriminator(self, embsize, n_classes):
        self.discriminator = AdversarialDiscriminator(
            d_model=embsize,
            n_cls=n_classes,
        ).to(self.device)

class scGPT(BaseMixin):
    """
    scGPT model
    Attributes:
        vocab: Vocabulary object
        model_config: dict
            {   layer_size: 128
                n_layers: 2
                nlayers: 4  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                nhead: 4  # number of heads in nn.MultiheadAttention
                dropout: 0.2  # dropout probability
                DSBN: False  # Domain-spec batchnorm
                fast_transformer: True
                freeze: False #freeze
            }
    """

    def __init__(self, vocab, pad_token, pad_value, num_batch_types, n_classes, model_config, freeze, device=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.model_config = model_config
        self.vocab = vocab
        self.ntokens = len(vocab)  # size of vocabulary
        self.freeze = freeze
        self.discriminator = None
        self.lr_ADV = None
        self.model = TransformerModel(
            self.ntokens,
            d_model=model_config.pop("embsize"),
            n_cls=n_classes,
            vocab=self.vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            num_batch_labels=num_batch_types,
            **model_config
        )

    def load_model(self, model_file):
        try:
            self.model.load_state_dict(torch.load(model_file))
            self.logger.info(f"Loading all model params from {model_file}")
        except:
            # only load params that are in the model and match the size
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                self.logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

    def freeze_model(self):
        pre_freeze_param_count = sum(
            dict((p.data_ptr(), p.numel()) for p in self.model.parameters() if p.requires_grad).values())

        # Freeze all pre-decoder weights
        for name, para in self.model.named_parameters():
            print("-" * 20)
            print(f"name: {name}")
            if self.freeze and "encoder" in name and "transformer_encoder" not in name:
                # if config.freeze and "encoder" in name:
                print(f"freezing weights for: {name}")
                para.requires_grad = False

        post_freeze_param_count = sum(
            dict((p.data_ptr(), p.numel()) for p in self.model.parameters() if p.requires_grad).values())

        self.logger.info(f"Total Pre freeze Params {(pre_freeze_param_count)}")
        self.logger.info(f"Total Post freeze Params {(post_freeze_param_count)}")


