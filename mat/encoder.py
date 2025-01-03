from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import x_transformers
from jaxtyping import Float
from torch import Tensor, nn

from mat.utils import init_weights


@dataclass
class EncoderConfig:
    obs_dim: int
    depth: int
    embed_dim: int
    num_heads: int
    num_agents: int


class EncoderOutput(NamedTuple):
    values: Float[Tensor, "b agents 1"]
    encoded: Float[Tensor, "b agents emb"]


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()

        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(config.obs_dim),
            nn.Linear(config.obs_dim, config.embed_dim),
            nn.GELU(),
        )
        self.obs_encoder.apply(partial(init_weights, use_relu_gain=True))

        self.encoder = x_transformers.Encoder(
            dim=config.embed_dim,
            depth=config.depth,
            heads=config.num_heads,
            dim_head=config.embed_dim // config.num_heads,
            max_seq_len=config.num_agents,
        )
        self.encoder.apply(partial(init_weights, use_relu_gain=True))

        self.value_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, 1),
        )
        self.value_head.apply(partial(init_weights, use_relu_gain=True))

    def forward(self, obs: Float[Tensor, "b agents obs"]) -> EncoderOutput:
        x = self.obs_encoder(obs)
        encoded = self.encoder(x)
        values = self.value_head(encoded)
        return EncoderOutput(values=values, encoded=encoded)
