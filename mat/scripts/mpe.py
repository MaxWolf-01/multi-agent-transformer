from __future__ import annotations

import argparse
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
from mat.paths import Paths
from mat.ppo_trainer import PPOTrainer, TrainerConfig
from mat.runners.mpe import MPERunner
from mat.samplers import DiscreteSampler, DiscreteSamplerConfig
from mat.scripts.config import ExperimentArgumentHandler, ExperimentConfig
from mat.utils import ModelCheckpointer, WandbArgumentHandler, WandbConfig


def main():
    cfg = get_config()
    encoder = mat.Encoder(cfg.encoder)
    decoder = mat.TransformerDecoder(cfg.decoder)
    sampler = DiscreteSampler(cfg.sampler)
    policy = MAT(encoder, decoder, sampler).to(cfg.device)
    buffer = Buffer(cfg.buffer)
    runner = MPERunner(
        env_id=cfg.env_id,
        env_kwargs=cfg.env_kwargs,
        policy=policy,
        buffer=buffer,
        num_envs=cfg.n_parallel_envs,
        device=cfg.device,
        render=cfg.render,
    )
    trainer = PPOTrainer(config=cfg.trainer, policy=policy, runner=runner)
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            config=asdict(cfg),
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
        )
    checkpointer = ModelCheckpointer(save_dir=Paths.CKPTS, prefix=cfg.wandb.name)
    pprint.pprint(cfg)

    steps_per_it = cfg.buffer.size * cfg.n_parallel_envs
    num_iterations = cfg.total_steps // steps_per_it
    total_steps = 0
    for i in range(num_iterations):
        metrics = trainer.train_iteration()
        total_steps += steps_per_it
        if i % cfg.log_every == 0:
            log_dict: dict[str, int | float] = {
                "iteration": i,
                "total_steps": total_steps,
                **asdict(metrics),
            }
            pprint.pprint(log_dict | dict(iteration=f"{i}/{num_iterations}"))
            print("-" * 40)
            if cfg.wandb.enabled:
                wandb.log(log_dict)
        if cfg.save_every is not None or cfg.save_best:
            checkpointer.save(
                model=policy, step=i, save_every_n=cfg.save_every, metric=metrics.mean_reward if cfg.save_best else None
            )


def get_config() -> MPEConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(MPERunner.SUPPORTED_ENVS.keys()), default="simple_spread_v3")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--save-every", type=int, help="Save model every n steps")
    parser.add_argument("--save-best", action="store_true", help="Save best model")
    parser.add_argument("--bs", type=int, help="Batch size")
    parser.add_argument("--episode-length", type=int, help="Length of an episode")
    ExperimentArgumentHandler.add_args(parser)
    WandbArgumentHandler.add_args(parser)
    args = parser.parse_args()

    cfg = MPEConfig.default(args.scenario)

    ExperimentArgumentHandler.update_config(args, cfg)
    WandbArgumentHandler.update_config(args, cfg)
    cfg.render = args.render or cfg.render
    cfg.n_parallel_envs = args.envs or cfg.n_parallel_envs
    cfg.buffer.num_envs = cfg.n_parallel_envs
    cfg.save_every = args.save_every or cfg.save_every
    cfg.save_best = args.save_best or cfg.save_best
    cfg.trainer.num_minibatches = args.bs or cfg.trainer.num_minibatches
    cfg.env_kwargs["max_cycles"] = args.episode_length or cfg.env_kwargs["max_cycles"]
    return cfg


@dataclass(kw_only=True, slots=True)
class MPEConfig(ExperimentConfig):
    env_id: str
    env_kwargs: dict[str, Any]
    render: bool
    save_every: int | None
    save_best: bool

    default_env_kwargs = dict(
        simple_spread_v3=dict(
            N=3,  # sets num landmarks and num agents
            max_cycles=25,
            continuous_actions=False,
            obs_dim=18,
            act_dim=5,
        ),
    )

    @classmethod
    def default(cls, scenario: str) -> MPEConfig:
        """Default config follows simple spread config from: https://github.com/PKU-MARL/Multi-Agent-Transformer/blob/e3cac1e39c2429f3cab93f2cbaca84481ac6539a/mat/scripts/train_mpe.sh"""
        env_kwargs = cls.default_env_kwargs[scenario]
        num_agents, episode_length, continuous_actions = 0, 0, False
        if scenario == "simple_spread_v3":
            num_agents = env_kwargs["N"]
            episode_length = env_kwargs["max_cycles"]
            continuous_actions = env_kwargs["continuous_actions"]
        obs_dim = env_kwargs.pop("obs_dim")
        act_dim = env_kwargs.pop("act_dim")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return MPEConfig(
            total_steps=20_000_000,
            n_parallel_envs=(n_parallel_envs := 128),
            log_every=10,
            device=device,
            env_id=scenario,
            env_kwargs=env_kwargs,
            render=False,
            save_every=None,
            save_best=True,
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
                start_token=1,
                device=device,
                dtype=torch.float32,
            ),
            buffer=BufferConfig(
                size=episode_length,  # episode length per original config, but not required to be the same
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
                num_ppo_epochs=10,
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
            wandb=WandbConfig(
                project=f"MPE_{scenario}",
            ),
        )


if __name__ == "__main__":
    main()
