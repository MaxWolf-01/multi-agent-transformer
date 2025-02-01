from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch

from mat.buffer import BufferConfig
from mat.config import DecoderType, EnvType, ExperimentArgumentHandler, ExperimentConfig, SamplerType
from mat.decoder import TransformerDecoderConfig
from mat.encoder import EncoderConfig
from mat.ppo_trainer import TrainerConfig
from mat.runners.mamujoco import MujocoRunnerConfig
from mat.samplers import ContinousSamplerConfig
from mat.train import train
from mat.utils import WandbArgumentHandler, WandbConfig


def main():
    train(get_config())


def get_config() -> MujocoConfig:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--scenario", type=str, help="HalfCheetah, Ant, etc.")
    parser.add_argument("--agent-conf", type=str, help="Agent configuration (e.g. '6x1', '2x3')")
    parser.add_argument("--agent-obsk", type=int, help="Agent observation depth")
    ExperimentArgumentHandler.add_args(parser)
    WandbArgumentHandler.add_args(parser)
    args = parser.parse_args()

    cfg = MujocoConfig.default_half_cheetah()

    ExperimentArgumentHandler.update_config(args, cfg)
    WandbArgumentHandler.update_config(args, cfg)
    cfg.runner.env_kwargs["agent_obsk"] = args.agent_obsk or cfg.runner.env_kwargs["agent_obsk"]
    return cfg


@dataclass(kw_only=True, slots=True)
class MujocoConfig(ExperimentConfig):
    runner: MujocoRunnerConfig

    @classmethod
    def default_half_cheetah(cls) -> MujocoConfig:
        """Default config follows original implementation: https://github.com/PKU-MARL/Multi-Agent-Transformer/blob/e3cac1e39c2429f3cab93f2cbaca84481ac6539a/mat/scripts/train_mujoco.sh"""
        scenario = "HalfCheetah"
        agent_conf = "6x1"
        num_agents = math.prod(map(int, agent_conf.split("x")))
        obs_dim = 7 + num_agents
        act_dim = 1  # max([space.shape[0] for space in env.action_spaces.values()])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return MujocoConfig(
            scenario=scenario,
            env_type=EnvType.MUJOCO,
            decoder_type=DecoderType.MAT,
            sampler_type=SamplerType.CONTINUOUS,
            total_steps=10_000_000,
            log_every=10,
            save_every=None,
            save_best=True,
            device=device,
            runner=MujocoRunnerConfig(
                env_kwargs=dict(
                    scenario=scenario, agent_conf=agent_conf, agent_obsk=0, max_episode_steps=(episode_length := 100)
                ),
                num_agents=num_agents,
                num_envs=(n_parallel_envs := 40),
                use_agent_id_enc=True,
                permute_agents=True,
                device=device,
                render=False,
            ),
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
                act_dim=act_dim,
                std_scale=0.5,
            ),
            buffer=BufferConfig(
                length=episode_length,  # episode length
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
                # num_minibatches=40,
                minibatch_size=100,  # OG: num_minibatch=40 => (40*100)/40 = 100
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
            wandb=WandbConfig(project=f"MaMuJoCo_{scenario}"),
        )


if __name__ == "__main__":
    main()
