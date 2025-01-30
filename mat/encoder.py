from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple

import x_transformers
from jaxtyping import Float
from torch import Tensor, nn
from x_transformers.x_transformers import AbsolutePositionalEmbedding, RotaryEmbedding

from mat.utils import init_weights


@dataclass
class EncoderConfig:
    obs_dim: int
    depth: int
    embed_dim: int
    num_heads: int
    pos_emb: Literal["absolute", "rotary"] | None
    pos_emb_kwargs: dict[str, int] | None


class EncoderOutput(NamedTuple):
    values: Float[Tensor, "b agents"]
    encoded: Float[Tensor, "b agents emb"]


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.obs_emb = nn.Sequential(
            nn.LayerNorm(config.obs_dim),
            nn.Linear(config.obs_dim, config.embed_dim),
            nn.GELU(),
        )
        self.obs_emb.apply(partial(init_weights, use_relu_gain=True))

        self.pos_emb = None
        self.use_rotary_pos_enc = config.pos_emb == "rotary"
        if config.pos_emb is not None:
            self.pos_emb = RotaryEmbedding if self.use_rotary_pos_enc else AbsolutePositionalEmbedding
            self.pos_emb = self.pos_emb(dim=config.embed_dim, **(config.pos_emb_kwargs or {}))

        self.encoder = x_transformers.Encoder(
            dim=config.embed_dim,
            depth=config.depth,
            heads=config.num_heads,
            attn_dim_head=config.embed_dim // config.num_heads,
            rotary_pos_emb=self.use_rotary_pos_enc,
        )
        self.encoder.apply(partial(init_weights, use_relu_gain=True))

        self.value_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, 1),
        )
        self.value_head.apply(partial(init_weights, use_relu_gain=True))

    def forward(self, obs: Float[Tensor, "b agents obs"], agent_perm: Float[Tensor, "agents"] | None = None) -> EncoderOutput:
        assert (agent_perm is None) == (self.pos_emb is None), "Pass agent_perm together with pos_emb or not at all."
        x = self.obs_emb(obs)
        if self.pos_emb is not None and not self.use_rotary_pos_enc:
            x += self.pos_emb(x[:, agent_perm, :])
        encoded = self.encoder(x, pos=agent_perm if self.use_rotary_pos_enc else None)
        values = self.value_head(encoded).squeeze()
        return EncoderOutput(values=values, encoded=encoded)
