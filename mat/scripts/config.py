import argparse
from dataclasses import dataclass


from mat.buffer import BufferConfig
from mat.decoder import (
    DecentralizedMlpDecoderConfig,
    TransformerDecoderConfig,
)
from mat.encoder import EncoderConfig
from mat.ppo_trainer import TrainerConfig
from mat.samplers import (
    ContinousSamplerConfig,
    DiscreteSamplerConfig,
)
from mat.utils import WandbArgumentHandler, WandbConfig


@dataclass(kw_only=True, slots=True)
class ExperimentConfig:
    encoder: EncoderConfig
    decoder: TransformerDecoderConfig | DecentralizedMlpDecoderConfig
    sampler: DiscreteSamplerConfig | ContinousSamplerConfig
    buffer: BufferConfig
    trainer: TrainerConfig
    wandb: WandbConfig
    n_parallel_envs: int
    total_steps: int
    log_every: int
    device: str


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, help="Episode length")
    parser.add_argument("--envs", type=int, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, help="Total number of steps to train")
    parser.add_argument("--log", type=int, help="Log frequency (in episodes)")
    WandbArgumentHandler.add_args(parser)


def update_config_from_args(parser: argparse.ArgumentParser, config: ExperimentConfig) -> ExperimentConfig:
    args = vars(parser.parse_args())
    WandbArgumentHandler.update_config(args, config)
    config.n_parallel_envs = args["envs"]
    config.total_steps = args["steps"]
    config.log_every = args["log"]
    return config
