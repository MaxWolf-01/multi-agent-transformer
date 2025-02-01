import argparse
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import wandb
from jaxtyping import Float
from torch import nn


class Paths:
    ROOT = Path(__file__).parent.parent
    PROJECT = ROOT / "src"
    VIDEOS = ROOT / "videos"
    CKPTS = ROOT / "checkpoints"
    DATA = ROOT / "data"
    LOGS = ROOT / "logs"


def init_weights(m: nn.Module, gain: float = 0.01, use_relu_gain: bool = False):
    if isinstance(m, nn.Linear):
        if use_relu_gain:
            gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def get_gae_returns_and_advantages(
    rewards: Float[np.ndarray, "steps envs agents"],
    values: Float[np.ndarray, "step envs agents"],
    dones: Float[np.ndarray, "steps envs agents"],
    gamma: float,
    gae_lambda: float,
) -> tuple[Float[np.ndarray, "steps envs agents"], Float[np.ndarray, "steps envs agents"]]:
    """Compute returns and advantages using Generalized Advantage Estimation (GAE)"""
    steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_gae = 0
    nonterminal_mask = 1 - dones
    for t in reversed(range(steps)):
        # TD error = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t + 1] * nonterminal_mask[t] - values[t]
        # GAE = sum (γλ)^k δ_{t+k} | Computed recursively: A_t = δ_t + γλA_{t+1}
        last_gae = advantages[t] = delta + gamma * gae_lambda * last_gae * nonterminal_mask[t]
    returns = advantages + values[:-1]
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
    enable: str = "wandb"
    project: str = "project"
    name: str = "name"
    tags: str = "tags"
    entity: str = "entity"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        wandb_group = parser.add_argument_group("Weights & Biases")
        wandb_group.add_argument(f"--{cls.enable}", action="store_true", help="Use Weights & Biases logger")
        wandb_group.add_argument(f"--{cls.project}", type=str, help="Name of the Weights & Biases project")
        wandb_group.add_argument(f"--{cls.name}", type=str, help="Name for the Weights & Biases run")
        wandb_group.add_argument(f"--{cls.tags}", nargs="+", help="Tags for the run. Example usage: --tags t1 t2 t3")
        wandb_group.add_argument(f"--{cls.entity}", type=str, help="Wandb entity")

    @classmethod
    def update_config(cls, namespace: argparse.Namespace, config: Any) -> None:
        args = vars(namespace)
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


class ModelCheckpointer:
    def __init__(self, save_dir: str | Path, prefix: str = ""):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.best_metric = float("-inf")
        self.best_path = None

    def save(
        self,
        model: nn.Module,
        step: int | None = None,
        save_every_n: int | None = None,
        metric: float | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Skips saving if:
        - Metric is provided and worse than the best metric seen
        - Step and save_every_n are provided, but current step is not a multiple of save_every_n
        """
        if (metric and metric < self.best_metric) or (step and save_every_n and step % save_every_n != 0):
            return
        state = {
            "model": model.state_dict(),
            "step": step,
            "metric": metric,
            "optimizer": optimizer.state_dict() if optimizer else None,
        }
        step_info = f"_{step}" if step else ""
        metric_info = f"_{metric:.3f}(best)" if metric else ""
        filepath = self.save_dir / f"{self.prefix}{step_info}{metric_info}.pt"
        torch.save(state, filepath)
        if metric and metric > self.best_metric:
            if self.best_path:
                self.best_path.unlink()
            self.best_path = filepath
            self.best_metric = metric


@dataclass
class HyperbandConfig:
    first_eval: int
    num_brackets: int
    eta: int

    @property
    def brackets(self) -> list[int]:
        return [self.first_eval * (self.eta**i) for i in range(self.num_brackets)]

    @property
    def max_iterations(self) -> int:
        return self.brackets[-1]


class WandbSweepConfig(TypedDict):
    method: str
    metric: dict[str, str]
    parameters: dict
    early_terminate: dict


def sweep_iteration(project: str, hyperband: HyperbandConfig, train_fn: callable) -> None:
    run = wandb.init(project=project)

    sweep_config = run.config
    args = []
    for key, value in sweep_config.items():
        if isinstance(value, bool) and value:
            args.append(f"--{key}")
        else:
            args.append(f"--{key}")
            args.append(str(value))
    args.extend(["--wandb", "--steps", str(hyperband.max_iterations * sweep_config["envs"] * sweep_config["buffer-len"])])
    sys.argv[1:] = args

    train_fn()


def run_sweep(project: str, sweep_params: dict, hyperband: HyperbandConfig, train_fn: callable) -> None:
    parser = argparse.ArgumentParser(description="Run or continue a W&B sweep")
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Existing sweep ID to continue / add an agent to a run. Format: username/project/sweep_id",
    )
    args = parser.parse_args()
    print(f"Training will use these evaluation checkpoints (iterations): {hyperband.brackets}")
    print(f"Final training length will be {hyperband.max_iterations} iterations")
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "mean_reward", "goal": "maximize"},
        "parameters": sweep_params,
        "early_terminate": {
            "type": "hyperband",
            "min_iter": hyperband.first_eval,
            "eta": hyperband.eta,
        },
    }
    sweep_id = args.sweep_id or wandb.sweep(sweep_config, project=project)
    wandb.agent(sweep_id, function=partial(sweep_iteration, project=project, hyperband=hyperband, train_fn=train_fn))
