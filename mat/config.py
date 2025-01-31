import argparse
import enum
from dataclasses import dataclass
from enum import auto

from mat.buffer import BufferConfig
from mat.decoder import (
    DecentralizedMlpDecoderConfig,
    TransformerDecoderConfig,
)
from mat.encoder import EncoderConfig
from mat.ppo_trainer import TrainerConfig
from mat.runners.mamujoco import MujocoRunnerConfig
from mat.runners.mpe import MPERunnerConfig
from mat.samplers import (
    ContinousSamplerConfig,
    DiscreteSamplerConfig,
)
from mat.utils import WandbConfig


class DecoderType(enum.StrEnum):
    MAT = auto()
    TRANSFORMER = auto()
    DECENTRALIZED = auto()


class SamplerType(enum.StrEnum):
    DISCRETE = auto()
    CONTINUOUS = auto()


class EnvType(enum.StrEnum):
    MPE = auto()
    MUJOCO = auto()


@dataclass(kw_only=True, slots=True)
class ExperimentConfig:
    env_type: EnvType
    scenario: str
    encoder: EncoderConfig
    decoder: TransformerDecoderConfig | DecentralizedMlpDecoderConfig
    decoder_type: DecoderType
    sampler: DiscreteSamplerConfig | ContinousSamplerConfig
    sampler_type: SamplerType
    buffer: BufferConfig
    runner: MPERunnerConfig | MujocoRunnerConfig
    trainer: TrainerConfig
    wandb: WandbConfig
    total_steps: int
    log_every: int
    save_every: int | None
    save_best: bool
    device: str


@dataclass
class ExperimentArgumentHandler:
    bs: str = "bs"
    lr: str = "lr"
    total_steps: str = "steps"
    ppo_epochs: str = "ppo-epochs"
    clip_param: str = "clip-param"
    value_loss_coef: str = "value-loss-coef"
    entropy_coef: str = "entropy-coef"
    max_grad_norm: str = "max-grad-norm"
    gamma: str = "gamma"
    gae_lambda: str = "gae-lambda"

    envs: str = "envs"
    episode_length: str = "episode-length"
    render: str = "render"
    buffer_len: str = "buffer-len"

    embed_dim: str = "embed-dim"
    n_heads: str = "n-heads"
    encoder_depth: str = "encoder-depth"
    decoder_depth: str = "decoder-depth"

    log_every: str = "log"
    save_every: str = "save-every"
    save_best: str = "save-best"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        training = parser.add_argument_group("Training Parameters")
        training.add_argument(f"--{cls.bs}", type=int, help="PPO mini-batch size")
        training.add_argument(f"--{cls.lr}", type=float, help="Learning rate")
        training.add_argument(f"--{cls.total_steps}", type=int, help="Total number of training steps")
        training.add_argument(f"--{cls.ppo_epochs}", type=int, help="Number of epochs over a trajectory")
        training.add_argument(f"--{cls.clip_param}", type=float, help="PPO clip parameter")
        training.add_argument(f"--{cls.value_loss_coef}", type=float, help="Value loss coefficient")
        training.add_argument(f"--{cls.entropy_coef}", type=float, help="Entropy coefficient")
        training.add_argument(f"--{cls.max_grad_norm}", type=float, help="Max gradient norm")
        training.add_argument(f"--{cls.gamma}", type=float, help="Discount factor")
        training.add_argument(f"--{cls.gae_lambda}", type=float, help="GAE lambda parameter")
        env = parser.add_argument_group("Environment Parameters")
        env.add_argument(f"--{cls.envs}", type=int, help="Number of parallel environments")
        env.add_argument(f"--{cls.episode_length}", type=int, help="Length of an episode")
        env.add_argument(f"--{cls.render}", action="store_true", help="Render the environment")
        env.add_argument(f"--{cls.buffer_len}", type=int, help="Horizon of the rollout buffer")
        model = parser.add_argument_group("Model Architecture Parameters")
        model.add_argument(f"--{cls.embed_dim}", type=int, help="Embedding dimension")
        model.add_argument(f"--{cls.n_heads}", type=int, help="Number of attention heads")
        model.add_argument(f"--{cls.encoder_depth}", type=int, help="Number of encoder blocks")
        model.add_argument(f"--{cls.decoder_depth}", type=int, help="Number of decoder blocks")
        logging = parser.add_argument_group("Logging and Checkpoint Parameters")
        logging.add_argument(f"--{cls.log_every}", type=int, help="Log frequency in iterations")
        logging.add_argument(f"--{cls.save_every}", type=int, help="Save model every n steps")
        logging.add_argument(f"--{cls.save_best}", action="store_true", help="Save best model")

    @classmethod
    def update_config(cls, namespace: argparse.Namespace, config: ExperimentConfig) -> None:
        args = {k.replace("_", "-"): v for k, v in vars(namespace).items()}

        config.trainer.minibatch_size = args[cls.bs] or config.trainer.minibatch_size
        config.trainer.lr = args[cls.lr] or config.trainer.lr
        config.trainer.num_ppo_epochs = args[cls.ppo_epochs] or config.trainer.num_ppo_epochs
        config.trainer.clip_param = args[cls.clip_param] or config.trainer.clip_param
        config.trainer.value_loss_coef = args[cls.value_loss_coef] or config.trainer.value_loss_coef
        config.trainer.entropy_coef = args[cls.entropy_coef] or config.trainer.entropy_coef
        config.trainer.max_grad_norm = args[cls.max_grad_norm] or config.trainer.max_grad_norm
        config.trainer.gamma = args[cls.gamma] or config.trainer.gamma
        config.trainer.gae_lambda = args[cls.gae_lambda] or config.trainer.gae_lambda

        config.buffer.num_envs = args[cls.envs] or config.buffer.num_envs
        config.runner.num_envs = args[cls.envs] or config.runner.num_envs
        config.buffer.length = args[cls.buffer_len] or config.buffer.length
        config.runner.render = args[cls.render] or config.runner.render
        eplen_kwarg = "max_cycles" if config.env_type == EnvType.MPE else "episode_length"
        config.runner.env_kwargs[eplen_kwarg] = args[cls.episode_length] or config.runner.env_kwargs[eplen_kwarg]

        config.encoder.embed_dim = args[cls.embed_dim] or config.encoder.embed_dim
        config.decoder.embed_dim = args[cls.embed_dim] or config.decoder.embed_dim
        config.encoder.num_heads = args[cls.n_heads] or config.encoder.num_heads
        config.decoder.num_heads = args[cls.n_heads] or config.decoder.num_heads
        config.encoder.depth = args[cls.encoder_depth] or config.encoder.depth
        config.decoder.depth = args[cls.decoder_depth] or config.decoder.depth

        config.total_steps = args[cls.total_steps] or config.total_steps
        config.log_every = args[cls.log_every] or config.log_every
        config.save_every = args[cls.save_every] or config.save_every
        config.save_best = args[cls.save_best] or config.save_best
