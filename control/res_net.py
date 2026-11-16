"""Wide residual MLP features extractor (SimBa / BRO style) for SB3.

All observation keys (or the ones listed in config) are flattened and joined,
projected to `width`, then passed through `blocks` pre-LayerNorm residual
blocks:  h = h + Dropout(W2 act(W1 LN(h))),  with W1: width -> expansion*width.
A final LayerNorm keeps the features at a fixed scale for the SB3 heads.
The last layer of every residual branch starts at zero, so the untrained net
is a plain projection and deep stacks train stably.
Settings in config/control.toml [res_net]; used by train.py --net res.
"""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResBlock(nn.Module):
    def __init__(self, width, expansion, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, expansion * width)
        self.down = nn.Linear(expansion * width, width)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, h):
        return h + self.drop(self.down(torch.relu(self.up(self.norm(h)))))


class ResMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, width=1024, blocks=4, expansion=4, dropout=0.01,
                 keys=()):
        super().__init__(observation_space, features_dim=width)
        self.keys = list(keys) or sorted(observation_space.spaces)
        n_in = sum(int(torch.tensor(observation_space[k].shape).prod()) for k in self.keys)
        self.inp = nn.Linear(n_in, width)
        self.blocks = nn.Sequential(*[ResBlock(width, expansion, dropout) for _ in range(blocks)])
        self.out_norm = nn.LayerNorm(width)

    def forward(self, obs):
        x = torch.cat([torch.flatten(obs[k], start_dim=1) for k in self.keys], dim=1)
        return self.out_norm(self.blocks(self.inp(x)))
