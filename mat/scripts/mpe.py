from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from mat.buffer import BufferConfig
from mat.config import DecoderType, EnvType, ExperimentArgumentHandler, ExperimentConfig, SamplerType
from mat.decoder import TransformerDecoderConfig
from mat.encoder import EncoderConfig
from mat.ppo_trainer import TrainerConfig
from mat.runners.mpe import MPERunnerConfig
from mat.samplers import DiscreteSamplerConfig
from mat.train import train
from mat.utils import WandbArgumentHandler, WandbConfig


def main():
    train(get_config())


def get_config() -> MPEConfig:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--scenario", choices=list(MPERunner.SUPPORTED_ENVS.keys()))
    parser.add_argument("--num-agents", type=int)
    ExperimentArgumentHandler.add_args(parser)
    WandbArgumentHandler.add_args(parser)
    args = parser.parse_args()

    cfg = MPEConfig.default_simple_spread()

    ExperimentArgumentHandler.update_config(args, cfg)
    WandbArgumentHandler.update_config(args, cfg)
    cfg.runner.env_kwargs["N"] = args.num_agents or cfg.runner.env_kwargs["N"]
    return cfg


@dataclass(kw_only=True, slots=True)
class MPEConfig(ExperimentConfig):
    runner: MPERunnerConfig

    @classmethod
    def default_simple_spread(cls) -> MPEConfig:
        """Default config follows simple spread config from: https://github.com/PKU-MARL/Multi-Agent-Transformer/blob/e3cac1e39c2429f3cab93f2cbaca84481ac6539a/mat/scripts/train_mpe.sh"""
        num_agents = 3
        obs_dim = 18 + num_agents  # add agent id encoding  # env.observation_space(env.aec_env.agents[0]).shape[0]
        act_dim = 5  # env.action_space(env.aec_env.agents[0]).n
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return MPEConfig(
            scenario="simple_spread_v3",
            env_type=EnvType.MPE,
            decoder_type=DecoderType.MAT,
            sampler_type=SamplerType.DISCRETE,
            total_steps=20_000_000,
            n_parallel_envs=(n_parallel_envs := 128),
            log_every=10,
            save_every=None,
            save_best=True,
            device=device,
            encoder=EncoderConfig(
                obs_dim=obs_dim,
                depth=1,
                embed_dim=64,
                num_heads=1,
            ),
            decoder=TransformerDecoderConfig(
                obs_dim=obs_dim,
                act_dim=act_dim,
                depth=2,
                embed_dim=64,
                num_heads=1,
                act_type="discrete",
                dec_actor=False,
            ),
            sampler=DiscreteSamplerConfig(
                act_dim=act_dim,
                start_token=1,
            ),
            buffer=BufferConfig(
                length=(episode_length := 25),
                num_envs=n_parallel_envs,
                num_agents=num_agents,
                obs_shape=(obs_dim,),
                action_dim=1,  # discrete == index -> (1,), continuous == value -> (act_dim,)
            ),
            runner=MPERunnerConfig(
                env_id="simple_spread_v3",
                env_kwargs=dict(
                    N=num_agents,  # sets num landmarks and num agents
                    max_cycles=episode_length,
                    continuous_actions=False,
                ),
                num_agents=num_agents,
                num_envs=n_parallel_envs,
                use_agent_id_enc=True,
                permute_agents=True,
                device=device,
                render=False,
            ),
            trainer=TrainerConfig(
                # optim
                lr=7e-4,
                eps=1e-5,
                weight_decay=0.0,
                max_grad_norm=0.5,
                # PPO
                num_ppo_epochs=10,
                minibatch_size=3200,  # OG: num_minibatch=1 => (128*25)/1 = 3200 batch size
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
            wandb=WandbConfig(project="MPE_simple_spread_v3"),
        )


if __name__ == "__main__":
    main()
