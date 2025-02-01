import pprint
from dataclasses import asdict

import wandb

from mat.buffer import Buffer
from mat.config import DecoderType, EnvType, ExperimentConfig, SamplerType
from mat.decoder import DecentralizedMlpDecoder, MATDecoder, TransformerDecoder
from mat.encoder import Encoder
from mat.mat import MAT
from mat.ppo_trainer import PPOTrainer
from mat.runners.mamujoco import MujocoRunner
from mat.runners.mpe import MPERunner
from mat.samplers import ContinuousSampler, DiscreteSampler
from mat.utils import ModelCheckpointer, Paths


def train(config: ExperimentConfig) -> None:
    pprint.pprint(config)
    encoder = Encoder(config.encoder)
    decoder = {
        DecoderType.MAT: MATDecoder,
        DecoderType.TRANSFORMER: TransformerDecoder,
        DecoderType.DECENTRALIZED: DecentralizedMlpDecoder,
    }[config.decoder_type](config.decoder)
    sampler = {
        SamplerType.DISCRETE: DiscreteSampler,
        SamplerType.CONTINUOUS: ContinuousSampler,
    }[config.sampler_type](config.sampler)
    policy = MAT(encoder, decoder, sampler).to(config.device)
    buffer = Buffer(config.buffer)
    runner = {
        EnvType.MPE: MPERunner,
        EnvType.MUJOCO: MujocoRunner,
    }[config.env_type](config.runner, policy=policy, buffer=buffer)
    trainer = PPOTrainer(config=config.trainer, policy=policy, runner=runner)
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            config=asdict(config),
            entity=config.wandb.entity,
            name=config.wandb.name,
            tags=config.wandb.tags,
        )
    checkpointer = ModelCheckpointer(save_dir=Paths.CKPTS, prefix=config.wandb.name)

    num_iterations = config.total_steps // buffer.batch_size
    total_steps = 0
    for i in range(num_iterations):
        metrics = trainer.train_iteration()
        total_steps += buffer.batch_size
        if i % config.log_every == 0:
            log_dict = {
                "iteration": i,
                "total_steps": total_steps,
                **asdict(metrics),
            }
            pprint.pprint(log_dict | dict(iteration=f"{i}/{num_iterations}"))
            print("-" * 40)
            if config.wandb.enabled:
                wandb.log(log_dict)
        if config.save_every is not None or config.save_best:
            checkpointer.save(
                model=policy, step=i, save_every_n=config.save_every, metric=metrics.mean_reward if config.save_best else None
            )
