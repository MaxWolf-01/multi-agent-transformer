import argparse
from dataclasses import dataclass

from mat.buffer import BufferConfig
from mat.decoder import (
    DecentralizedMlpDecoderConfig,
    TransformerDecoderConfig,
)
from mat.encoder import EncoderConfig
from mat.ppo_trainer import TrainerConfig
from mat.runners.base import RunnerConfig
from mat.samplers import (
    ContinousSamplerConfig,
    DiscreteSamplerConfig,
)
from mat.utils import WandbConfig


@dataclass(kw_only=True, slots=True)
class ExperimentConfig:
    encoder: EncoderConfig
    decoder: TransformerDecoderConfig | DecentralizedMlpDecoderConfig
    sampler: DiscreteSamplerConfig | ContinousSamplerConfig
    buffer: BufferConfig
    runner: RunnerConfig
    trainer: TrainerConfig
    wandb: WandbConfig
    n_parallel_envs: int
    total_steps: int
    log_every: int
    device: str


@dataclass
class ExperimentArgumentHandler:
    bs: str = "bs"
    lr: str = "lr"
    envs: str = "envs"
    steps: str = "steps"
    log_every: str = "log"
    ppo_epochs: str = "ppo-epochs"
    buffer_len: str = "buffer-len"
    render: str = "render"
    save_every: str = "save-every"
    save_best: str = "save-best"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(f"--{cls.bs}", type=int, help="PPO mini-batch size")
        parser.add_argument(f"--{cls.lr}", type=float, help="Learning rate")
        parser.add_argument(f"--{cls.envs}", type=int, help="Number of parallel environments")
        parser.add_argument(f"--{cls.steps}", type=int, help="Total number of training steps")
        parser.add_argument(f"--{cls.log_every}", type=int, help="Log frequency in iterations")
        parser.add_argument(f"--{cls.ppo_epochs}", type=int, help="Number of epochs over a trajectory")
        parser.add_argument(f"--{cls.buffer_len}", type=int, help="Horizon of the rollout buffer")
        parser.add_argument(f"--{cls.render}", action="store_true", help="Render the environment")
        parser.add_argument(f"--{cls.save_every}", type=int, help="Save model every n steps")
        parser.add_argument(f"--{cls.save_best}", action="store_true", help="Save best model")

    @classmethod
    def update_config(cls, namespace: argparse.Namespace, config: ExperimentConfig) -> None:
        args = {k.replace("_", "-"): v for k, v in vars(namespace).items()}  # argparse converts hyphens to underscores
        config.trainer.minibatch_size = args[cls.bs] or config.trainer.minibatch_size
        config.trainer.lr = args[cls.lr] or config.trainer.lr
        config.n_parallel_envs = args[cls.envs] or config.n_parallel_envs
        config.buffer.num_envs = args[cls.envs] or config.n_parallel_envs
        config.total_steps = args[cls.steps] or config.total_steps
        config.log_every = args[cls.log_every] or config.log_every
        config.trainer.num_ppo_epochs = args[cls.ppo_epochs] or config.trainer.num_ppo_epochs
        config.buffer.length = args[cls.buffer_len] or config.buffer.length
