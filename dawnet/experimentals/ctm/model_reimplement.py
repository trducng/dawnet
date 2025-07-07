import torch
import torch.nn as nn


class CTM(nn.Module):
    """Reimplementation of the CTM module"""
    def __init__(self, size, memory, nticks, **config):
        super().__init__()

        self._config = config
        self._memory = memory
        self._nticks = nticks

        self._feature_encoder = self.get_feature_encoder()
        self._init_z = nn.Parameter(torch.empty(size), requires_grad=True)
        self._init_nlm = ...


    def get_feature_encoder(self) -> nn.Module:
        """Encode the feature"""
        ...


    def forward(self, x):
        feat = self._feature_encoder(x)
