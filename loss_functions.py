import torch.nn as nn
import torch
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.model import TransformerModel, AdversarialDiscriminator

class BaseMixin:
    def __init__(self, lr, schedule_interval, schedule_ratio, DAB_separate_optim, ADV, amp, lr_ADV=None, discriminator=None):
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

class scGPT(BaseMixin, nn.Module):
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
    def __init__(self, vocab, model_config, device=None, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.model_config = model_config
        self.vocab = vocab
        self.ntokens = len(vocab)  # size of vocabulary
        self.model = TransformerModel(
            self.ntokens,
            embsize,
            nhead,
            d_hid,
            nlayers,
            nlayers_cls=3,
            n_cls=num_types if CLS else 1,
            vocab=self.vocab,
            dropout=dropout,
            pad_token=pad_token,
            pad_value=pad_value,
            do_mvc=MVC,
            do_dab=DAB,
            use_batch_labels=INPUT_BATCH_LABELS,
            num_batch_labels=num_batch_types,
            domain_spec_batchnorm=config.DSBN,
            input_emb_style=input_emb_style,
            n_input_bins=n_input_bins,
            cell_emb_style=cell_emb_style,
            mvc_decoder_style=mvc_decoder_style,
            ecs_threshold=ecs_threshold,
            explicit_zero_prob=explicit_zero_prob,
            use_fast_transformer=fast_transformer,
            fast_transformer_backend=fast_transformer_backend,
            pre_norm=config.pre_norm,
            **model_config
        )

        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        with torch.cuda.amp.autocast(enabled=self.amp):
            y_hat = self.model(x)
            loss = self.losses['default'](y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.cuda.amp.autocast(enabled=self.amp):
            y_hat = self.model(x)
            loss = self.losses['default'](y_hat, y)
        return loss

    def configure_optimizers(self):
        return self.optimizers['optimizer'], self.lr_schedulers['scheduler']

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        return {'loss': avg_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        return {'val_loss': avg_loss}