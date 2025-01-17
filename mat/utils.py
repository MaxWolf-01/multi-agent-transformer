import argparse
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from jaxtyping import Float
from torch import nn


def init_weights(m: nn.Module, gain: float = 0.01, use_relu_gain: bool = False):
    if isinstance(m, nn.Linear):
        if use_relu_gain:
            gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def get_gae_returns_and_advantages(
    rewards: Float[np.ndarray, "steps envs agents"],
    values: Float[np.ndarray, "steps envs agents"],
    next_value: Float[np.ndarray, "envs agents"],
    gamma: float,
    gae_lambda: float,
) -> tuple[Float[np.ndarray, "steps envs agents"], Float[np.ndarray, "steps envs agents"]]:
    """Compute returns and advantages using Generalized Advantage Estimation (GAE)"""
    steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_advantage = np.zeros_like(next_value)

    for t in reversed(range(steps)):
        if t == steps - 1:
            next_value_t = next_value
        else:
            next_value_t = values[t + 1]
        # TD error = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value_t - values[t]
        # GAE = sum (γλ)^k δ_{t+k} | Computed recursively: A_t = δ_t + γλA_{t+1}
        advantages[t] = delta + gamma * gae_lambda * last_advantage
        last_advantage = advantages[t]
    returns = advantages + values
    return returns, advantages


@dataclass
class WandbConfig:
    id: str | None = None
    enabled: bool = False
    project: str | None = None
    name: str | None = None
    tags: list[str] | None = None
    entity: str | None = None


@dataclass
class WandbArgumentHandler:
    """Handles argument parsing and configuration for wandb logging"""

    enable: str = "wandb"
    project: str = "project"
    name: str = "name"
    tags: str = "tags"
    entity: str = "entity"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(f"--{cls.enable}", action="store_true", help="Use Weights & Biases logger")
        parser.add_argument(f"--{cls.project}", type=str, help="Name of the Weights & Biases project")
        parser.add_argument(f"--{cls.name}", type=str, help="Name for the Weights & Biases run")
        parser.add_argument(f"--{cls.tags}", nargs="+", help="Tags for the run. Example usage: --tags t1 t2 t3")
        parser.add_argument(f"--{cls.entity}", type=str, help="Wandb entity")

    @classmethod
    def update_config(cls, args: dict[str, Any], config: Any) -> None:
        config.wandb.enabled = args[cls.enable]
        config.wandb.project = args[cls.project] or config.wandb.project
        config.wandb.name = get_run_name(args[cls.name] or config.wandb.name)
        config.wandb.tags = args[cls.tags] or config.wandb.tags
        config.wandb.entity = args[cls.entity] or config.wandb.entity


def get_run_name(name: str | None = None) -> str:
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    base_name = name if name is not None else f"no_name_{short_uuid(4)}"
    return f"{base_name}-{timestamp}"


def short_uuid(n: int = 8) -> str:
    return str(uuid.uuid4())[:n]
