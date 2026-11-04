"""3-layer MLP on the raw ToF readings only, with LayerNorm and dropout.

Used as the SB3 features extractor: it reads obs["tof"] (sensor x history x
field, see bop_env.env.TOF_FIELDS) and ignores every other observation key.
The SB3 heads on top map the features to the 3 servo positions (actor) and
to the value / Q estimate (critic). Settings in config/control.toml [tof_net].
"""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ToFNet(BaseFeaturesExtractor):
    def __init__(self, observation_space, layers=(256, 256, 256), dropout=0.1):
        super().__init__(observation_space, features_dim=layers[-1])
        n_in = int(torch.tensor(observation_space["tof"].shape).prod())
        blocks = []
        for n_out in layers:  # Linear -> LayerNorm -> ReLU -> Dropout
            blocks += [nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.ReLU(), nn.Dropout(dropout)]
            n_in = n_out
        self.net = nn.Sequential(*blocks)

    def forward(self, obs):
        return self.net(torch.flatten(obs["tof"], start_dim=1))
