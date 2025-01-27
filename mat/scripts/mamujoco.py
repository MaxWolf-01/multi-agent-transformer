from __future__ import annotations

import argparse
import math
import pprint
from dataclasses import asdict, dataclass
from typing import Any

import torch
import wandb
from gymnasium_robotics.envs.multiagent_mujoco.mujoco_multi import parallel_env

from mat import mat
from mat.buffer import Buffer, BufferConfig
from mat.decoder import TransformerDecoderConfig
from mat.encoder import EncoderConfig
from mat.mat import MAT
from mat.paths import Paths
from mat.ppo_trainer import PPOTrainer, TrainerConfig
from mat.runners.mamujoco import MujocoRunner
from mat.samplers import ContinousSamplerConfig, ContinuousSampler
from mat.scripts.config import ExperimentArgumentHandler, ExperimentConfig
from mat.utils import ModelCheckpointer, WandbArgumentHandler, WandbConfig


def main():
    cfg = get_config()
    encoder = mat.Encoder(cfg.encoder)
    decoder = mat.TransformerDecoder(cfg.decoder)
    sampler = ContinuousSampler(cfg.sampler)
    policy = MAT(encoder, decoder, sampler).to(cfg.device)
    buffer = Buffer(cfg.buffer)
    print(cfg.env_kwargs)
    runner = MujocoRunner(
        # episode_length=cfg.episode_length,
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


def get_config() -> MujocoConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, help="HalfCheetah, Ant, etc.")
    parser.add_argument("--agent-conf", type=str, help="Agent configuration (e.g. '6x1', '2x3')")
    parser.add_argument("--agent-obsk", type=int, help="Agent observation depth")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--save-every", type=int, help="Save model every n steps")
    parser.add_argument("--save-best", action="store_true", help="Save best model")
    parser.add_argument("--bs", type=int, help="Batch size")
    # parser.add_argument("--episode-length", type=int, help="Length of an episode")
    ExperimentArgumentHandler.add_args(parser)
    WandbArgumentHandler.add_args(parser)
    args = parser.parse_args()

    cfg = MujocoConfig.default(args.scenario, args.agent_conf)

    ExperimentArgumentHandler.update_config(args, cfg)
    WandbArgumentHandler.update_config(args, cfg)
    cfg.render = args.render or cfg.render
    cfg.n_parallel_envs = args.envs or cfg.n_parallel_envs
    cfg.buffer.num_envs = cfg.n_parallel_envs
    cfg.save_every = args.save_every or cfg.save_every
    cfg.save_best = args.save_best or cfg.save_best
    cfg.trainer.num_minibatches = args.bs or cfg.trainer.num_minibatches
    cfg.env_kwargs["agent_obsk"] = args.agent_obsk or cfg.env_kwargs["agent_obsk"]
    return cfg


@dataclass(kw_only=True, slots=True)
class MujocoConfig(ExperimentConfig):
    env_kwargs: dict[str, Any]
    # episode_length: int
    render: bool
    save_every: int | None
    save_best: bool

    @classmethod
    def default(cls, scenario: str | None, agent_conf: str | None) -> MujocoConfig:
        """Default config follows original implementation: https://github.com/PKU-MARL/Multi-Agent-Transformer/blob/e3cac1e39c2429f3cab93f2cbaca84481ac6539a/mat/scripts/train_mujoco.sh"""
        scenario = scenario or "HalfCheetah"
        agent_conf = agent_conf or "6x1"
        env = parallel_env(scenario=scenario, agent_conf=agent_conf)
        obs, _ = env.reset()
        obs_dim = 7  # len(env.map_local_observations_to_global_state(obs)) # TODO get the actual thing programmatically
        act_dim = max([space.shape[0] for space in env.action_spaces.values()])
        num_agents = math.prod(map(int, agent_conf.split("x")))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return MujocoConfig(
            total_steps=10_000_000,
            n_parallel_envs=(n_parallel_envs := 40),
            log_every=10,
            device=device,
            # episode_length=1000,
            env_kwargs=dict(
                scenario=scenario,
                agent_conf=agent_conf,
                agent_obsk=0,
            ),
            render=False,
            save_every=None,
            save_best=True,
            encoder=EncoderConfig(
                obs_dim=obs_dim,
                depth=1,
                embed_dim=64,
                num_heads=1,
            ),
            decoder=TransformerDecoderConfig(
                obs_dim=obs_dim,
                act_dim=act_dim,
                depth=1,
                embed_dim=64,
                num_heads=1,
                act_type="continuous",
                dec_actor=False,
            ),
            sampler=ContinousSamplerConfig(
                num_agents=num_agents,
                act_dim=act_dim,
                device=device,
                dtype=torch.float32,
                std_scale=0.5,
            ),
            buffer=BufferConfig(
                size=1000,  # episode length
                num_envs=n_parallel_envs,
                num_agents=num_agents,
                obs_shape=(obs_dim,),
                action_dim=act_dim,
            ),
            trainer=TrainerConfig(
                # optim
                lr=5e-5,
                eps=1e-5,
                weight_decay=0.0,
                max_grad_norm=0.5,
                # PPO
                num_ppo_epochs=10,
                num_minibatches=40,
                clip_param=0.05,
                value_loss_coef=1.0,
                entropy_coef=0.001,
                gamma=0.99,
                gae_lambda=0.95,
                use_clipped_value_loss=True,
                normalize_advantage=True,
                use_huber_loss=True,
                huber_delta=10.0,
                device=device,
            ),
            wandb=WandbConfig(
                project=f"MaMuJoCo_{scenario}",
            ),
        )


if __name__ == "__main__":
    main()
