from __future__ import annotations

import pprint
from dataclasses import asdict, dataclass
from typing import Any

import torch
import wandb

from mat import mat
from mat.buffer import Buffer, BufferConfig
from mat.decoder import TransformerDecoderConfig
from mat.encoder import EncoderConfig
from mat.mat import MAT
from mat.ppo_trainer import PPOTrainer, TrainerConfig
from mat.runners.mpe import MPERunner
from mat.samplers import DiscreteSampler, DiscreteSamplerConfig
from mat.scripts.config import ExperimentConfig
from mat.utils import WandbConfig


@dataclass(kw_only=True, slots=True)
class MPEConfig(ExperimentConfig):
    env_config: dict[str, Any]

    @classmethod
    def default(cls, scenario: str) -> MPEConfig:
        """Default config follows simple spread config from: https://github.com/PKU-MARL/Multi-Agent-Transformer/blob/e3cac1e39c2429f3cab93f2cbaca84481ac6539a/mat/scripts/train_mpe.sh"""
        default_env_kwargs = dict(
            simple_spread_v3=dict(
                N=(num_agents := 3),  # sets num landmarks and num agents
                max_cycles=(episode_length := 25),
                continuous_actions=(continuous_actions := False),
                obs_dim=18,
                act_dim=5,
            ),
        )
        obs_dim = default_env_kwargs[scenario].pop("obs_dim")
        act_dim = default_env_kwargs[scenario].pop("act_dim")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return MPEConfig(
            total_steps=20_000_000,
            n_parallel_envs=(n_parallel_envs := 128),
            log_every=10,
            device=device,
            env_config=dict(env_id=scenario) | default_env_kwargs[scenario],
            encoder=EncoderConfig(
                obs_dim=obs_dim,  # env.observation_space(env.aec_env.agents[0]).shape[0]
                depth=1,
                embed_dim=64,
                num_heads=1,
            ),
            decoder=TransformerDecoderConfig(
                obs_dim=obs_dim,
                act_dim=act_dim,  # env.action_space(env.aec_env.agents[0]).n
                depth=2,
                embed_dim=64,
                num_heads=1,
                num_agents=num_agents,  # len(env.aec_env.agents)
                act_type="continuous" if continuous_actions else "discrete",
                dec_actor=False,
            ),
            sampler=DiscreteSamplerConfig(
                num_agents=num_agents,
                act_dim=act_dim,
                start_token=1,  # TODO doesn't this overlap with the action space?
                device=device,
                dtype=torch.float32,
            ),
            buffer=BufferConfig(
                size=episode_length,
                num_envs=n_parallel_envs,
                num_agents=num_agents,
                obs_shape=(obs_dim,),
                action_dim=act_dim if continuous_actions else 1,  # discrete == index, continuous == value
            ),
            trainer=TrainerConfig(
                # optim
                lr=7e-4,
                eps=1e-5,
                weight_decay=0.0,
                max_grad_norm=0.5,
                # PPO
                num_epochs=10,
                num_minibatches=1,
                clip_param=0.05,
                value_loss_coef=1.0,
                entropy_coef=0.01,
                gamma=0.99,
                gae_lambda=0.95,
                use_clipped_value_loss=True,
                normalize_advantage=True,
                use_huber_loss=True,
                huber_delta=10.0,
                device=device,
            ),
            wandb=WandbConfig(),
        )


def main():
    cfg = MPEConfig.default("simple_spread_v3")
    encoder = mat.Encoder(cfg.encoder)
    decoder = mat.TransformerDecoder(cfg.decoder)
    sampler = DiscreteSampler(cfg.sampler)
    policy = MAT(encoder, decoder, sampler).to(cfg.device)
    buffer = Buffer(cfg.buffer)
    runner = MPERunner(
        env_id=cfg.env_config.pop("env_id"),
        env_kwargs=cfg.env_config,
        policy=policy,
        buffer=buffer,
        num_envs=cfg.n_parallel_envs,
        device=cfg.device,
        render=True,
    )
    trainer = PPOTrainer(config=cfg.trainer, policy=policy, runner=runner)

    steps_per_episode = cfg.buffer.size * cfg.n_parallel_envs
    num_episodes = cfg.total_steps // steps_per_episode
    total_steps = 0
    for episode in range(num_episodes):
        metrics = trainer.train_episode()
        total_steps += steps_per_episode
        if episode % cfg.log_every == 0:
            log_dict = {
                "episode": episode,
                "total_steps": total_steps,
                **asdict(metrics),
            }
            pprint.pprint(log_dict)
            print("-" * 40)
            if cfg.wandb.enabled:
                wandb.log(log_dict)


if __name__ == "__main__":
    main()
