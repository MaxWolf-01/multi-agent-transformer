from __future__ import annotations

import argparse
import pprint
from dataclasses import asdict, dataclass

import torch
import wandb

from mat import mat
from mat.buffer import Buffer, BufferConfig
from mat.decoder import TransformerDecoderConfig
from mat.encoder import EncoderConfig
from mat.mat import MAT
from mat.paths import Paths
from mat.ppo_trainer import PPOTrainer, TrainerConfig
from mat.runners.mpe import MPERunner, MPERunnerConfig
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
    runner = MPERunner(cfg.runner, buffer=buffer, policy=policy)
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

    steps_per_it = cfg.buffer.length * cfg.n_parallel_envs
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
    parser.add_argument("--episode-length", type=int, help="Length of an episode")
    ExperimentArgumentHandler.add_args(parser)
    WandbArgumentHandler.add_args(parser)
    args = parser.parse_args()

    cfg = MPEConfig.default(args.scenario)

    ExperimentArgumentHandler.update_config(args, cfg)
    WandbArgumentHandler.update_config(args, cfg)
    cfg.runner.env_kwargs["max_cycles"] = args.episode_length or cfg.runner.env_kwargs["max_cycles"]
    return cfg


@dataclass(kw_only=True, slots=True)
class MPEConfig(ExperimentConfig):
    runner: MPERunnerConfig

    @classmethod
    def default(cls, scenario: str) -> MPEConfig:
        """Default config follows simple spread config from: https://github.com/PKU-MARL/Multi-Agent-Transformer/blob/e3cac1e39c2429f3cab93f2cbaca84481ac6539a/mat/scripts/train_mpe.sh"""
        env_kwargs = dict(
            simple_spread_v3=dict(
                N=3,  # sets num landmarks and num agents
                max_cycles=25,
                continuous_actions=False,
                obs_dim=18,  # env.observation_space(env.aec_env.agents[0]).shape[0]
                act_dim=5,  # env.action_space(env.aec_env.agents[0]).n
            ),
        )[scenario]
        num_agents, episode_length, continuous_actions = 0, 0, False
        if scenario == "simple_spread_v3":
            num_agents = env_kwargs["N"]
            episode_length = env_kwargs["max_cycles"]
            continuous_actions = env_kwargs["continuous_actions"]
        obs_dim = env_kwargs.pop("obs_dim") + num_agents  # add agent id encoding
        act_dim = env_kwargs.pop("act_dim")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return MPEConfig(
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
                pos_emb=None,  # "absolute",
                pos_emb_kwargs=None,  # {"max_seq_len": num_agents},
            ),
            decoder=TransformerDecoderConfig(
                obs_dim=obs_dim,
                act_dim=act_dim,
                depth=2,
                embed_dim=64,
                num_heads=1,
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
                length=episode_length,  # episode length per original config, but not required to be the same
                num_envs=n_parallel_envs,
                num_agents=num_agents,
                obs_shape=(obs_dim,),
                action_dim=act_dim if continuous_actions else 1,  # discrete == index, continuous == value
            ),
            runner=MPERunnerConfig(
                env_id=scenario,
                env_kwargs=env_kwargs,
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
            wandb=WandbConfig(
                project=f"MPE_{scenario}",
            ),
        )


if __name__ == "__main__":
    main()
