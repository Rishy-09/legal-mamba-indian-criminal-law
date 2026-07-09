import torch
import torch.nn as nn

from mamba import Mamba, ModelArgs


class BNSMambaConfig:
    def __init__(self, vocab_size, d_model=512, n_layers=24):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers


class BNSMambaModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        args = ModelArgs(
            d_model=config.d_model,
            n_layer=config.n_layers,
            vocab_size=config.vocab_size,
            d_state=16,
            expand=2,
            d_conv=4,
        )

        self.model = Mamba(args)

    def forward(self, x):
        return self.model(x)