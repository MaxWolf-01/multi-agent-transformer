from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
import x_transformers
from jaxtyping import Float
from torch import Tensor, nn

from mat.utils import init_weights


@dataclass
class TransformerDecoderConfig:
    obs_dim: int
    act_dim: int
    depth: int
    embed_dim: int
    num_heads: int
    act_type: Literal["discrete", "continuous"]
    dec_actor: bool


class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerDecoderConfig) -> None:
        super().__init__()
        self.cfg = config
        self.act_emb = nn.Sequential(
            nn.Linear(
                config.act_dim if config.act_type != "discrete" else config.act_dim + 1,
                config.embed_dim,
                config.act_type != "discrete",
            ),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim),
        )
        self.act_emb.apply(partial(init_weights, use_relu_gain=True))
        self.decoder = x_transformers.Decoder(
            dim=config.embed_dim,
            depth=config.depth,
            heads=config.num_heads,
            attn_dim_head=config.embed_dim // config.num_heads,
            cross_attend=True,
        )
        self.decoder.apply(partial(init_weights, use_relu_gain=True))
        self.act_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Linear(config.embed_dim, config.act_dim),
        )
        self.act_head.apply(partial(init_weights, use_relu_gain=True))
        self.log_std = nn.Parameter(torch.ones(config.act_dim)) if config.act_type == "continuous" else None

    def forward(
        self, action: Float[Tensor, "b agents act"], encoded_obs: Float[Tensor, "b agents dim"]
    ) -> Float[Tensor, "b agents act"]:
        x = self.act_emb(action)
        x = self.decoder(
            x=x,
            context=encoded_obs,
            context_mask=None,  # TODO
            in_attn_cond=encoded_obs,  # adds encoded_obs residual after cross-attention
        )
        return self.act_head(x)  # TODO allow for multi


@dataclass
class DecentralizedMlpDecoderConfig:
    obs_dim: int
    act_dim: int
    depth: int
    embed_dim: int
    num_heads: int
    num_agents: int
    act_type: Literal["discrete", "continuous"]
    dec_actor: bool
    shared_actor: bool


class DecentralizedMlpDecoder(nn.Module):
    def __init__(self, config: DecentralizedMlpDecoderConfig) -> None:
        super().__init__()
        self.cfg = config

        def get_mlp():
            mlp = nn.Sequential(
                nn.LayerNorm(config.obs_dim),
                nn.Linear(config.obs_dim, config.embed_dim),
                nn.GELU(),
                nn.LayerNorm(config.embed_dim),
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.GELU(),
                nn.LayerNorm(config.embed_dim),
                nn.Linear(config.embed_dim, config.act_dim),
            )
            mlp.apply(partial(init_weights, use_relu_gain=True))
            return mlp

        self.mlp = get_mlp() if config.shared_actor else nn.ModuleList([get_mlp() for _ in range(config.num_agents)])
        self.log_std = nn.Parameter(torch.ones(config.act_dim)) if config.act_type == "continuous" else None

    def forward(self, encoded_obs: Float[Tensor, "b agents dim"]) -> Float[Tensor, "b agents act"]:
        if self.cfg.shared_actor:
            return self.mlp(encoded_obs)
        return torch.stack([mlp(encoded_obs[:, i]) for i, mlp in enumerate(self.mlp)], dim=1)


class MATDecoderLayer(nn.Module):
    def __init__(self, heads: int, embed_dim: int) -> None:
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.self_attn = x_transformers.Attention(
            heads=heads,
            dim=embed_dim,
            dim_head=embed_dim // heads,
            causal=True,
        )
        self.cross_attn = x_transformers.Attention(
            heads=heads,
            dim=embed_dim,
            dim_head=embed_dim // heads,
            causal=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mlp.apply(partial(init_weights, use_relu_gain=True))

    def forward(self, x: Float[Tensor, "b agents dim"], context: Float[Tensor, "b agents dim"]) -> Float[Tensor, "b agents dim"]:
        x = self.ln1(x + self.self_attn(x))
        x = self.ln2(context + self.cross_attn(context, context=x))
        return self.ln3(x + self.mlp(x))


class MATDecoder(nn.Module):
    """Decoder that exactly follows the MAT paper and implementation: Switched cross-attn sources."""

    def __init__(self, config: TransformerDecoderConfig) -> None:
        super().__init__()
        self.cfg = config
        self.act_emb = nn.Sequential(
            nn.Linear(
                config.act_dim if config.act_type != "discrete" else config.act_dim + 1,
                config.embed_dim,
                config.act_type != "discrete",
            ),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim),
        )
        self.act_emb.apply(partial(init_weights, use_relu_gain=True))
        self.decoder = nn.ModuleList([MATDecoderLayer(config.num_heads, config.embed_dim) for _ in range(config.depth)])
        self.act_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Linear(config.embed_dim, config.act_dim),
        )
        self.act_head.apply(partial(init_weights, use_relu_gain=True))
        self.log_std = nn.Parameter(torch.ones(config.act_dim)) if config.act_type == "continuous" else None

    def forward(
        self, action: Float[Tensor, "b agents act"], encoded_obs: Float[Tensor, "b agents dim"]
    ) -> Float[Tensor, "b agents act"]:
        x = self.act_emb(action)
        for layer in self.decoder:
            x = layer(x, context=encoded_obs)
        return self.act_head(x)
