from dataclasses import dataclass

import torch

from mat import mat
from mat.buffer import Buffer, BufferConfig
from mat.decoder import TransformerDecoderConfig
from mat.encoder import EncoderConfig
from mat.mat import MAT
from mat.ppo_trainer import PPOTrainer, TrainerConfig
from mat.runners.mpe import MPERunner
from mat.samplers import DiscreteSampler, DiscreteSamplerConfig


@dataclass
class RunConfig:
    n_parallel_envs: int = 32
    total_steps: int = 10_000_000
    log_every: int = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    run_cfg = RunConfig()

    env = MPERunner.get_env(
        env_id="simple_spread_v3",
        env_kwargs=dict(
            N=3,  # num landmarks, num agents
            max_cycles=(episode_length := 25),
            continuous_actions=False,
        ),
    )
    env.reset()
    obs_dim = env.observation_space(env.aec_env.agents[0]).shape[0]
    act_dim = env.action_space(env.aec_env.agents[0]).n
    num_agents = len(env.aec_env.agents)

    encoder_cfg = EncoderConfig(
        obs_dim=obs_dim,
        depth=2,
        embed_dim=64,
        num_heads=1,
    )

    decoder_cfg = TransformerDecoderConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        depth=2,
        embed_dim=64,
        num_heads=1,
        num_agents=num_agents,
        act_type="discrete",  # mpe can be both
        dec_actor=False,
    )

    sampler_cfg = DiscreteSamplerConfig(
        num_agents=num_agents,
        act_dim=act_dim,
        start_token=1,
        device=run_cfg.device,
        dtype=torch.float32,
    )

    buffer_cfg = BufferConfig(
        size=episode_length,
        num_envs=run_cfg.n_parallel_envs,
        num_agents=num_agents,
        obs_shape=(obs_dim,),
        action_dim=1,  # discrete == index
    )

    trainer_cfg = TrainerConfig(
        # optim
        lr=5e-4,
        eps=1e-5,
        weight_decay=0.0,
        max_grad_norm=0.5,
        # PPO
        num_epochs=10,
        num_minibatches=4,
        clip_param=0.2,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        use_clipped_value_loss=True,
        normalize_advantage=True,
        use_huber_loss=True,
        huber_delta=10.0,
        device=run_cfg.device,
    )

    encoder = mat.Encoder(encoder_cfg)
    decoder = mat.TransformerDecoder(decoder_cfg)
    sampler = DiscreteSampler(sampler_cfg)
    policy = MAT(
        encoder=encoder,
        decoder=decoder,
        sampler=sampler,
    ).to(run_cfg.device)
    buffer = Buffer(buffer_cfg)
    # runner = MPERunner(num_agents=num_agents, device, env, policy, buffer)
    runner = MPERunner(env, policy, buffer, num_envs=run_cfg.n_parallel_envs, device=run_cfg.device)
    trainer = PPOTrainer(trainer_cfg, policy, runner)

    num_episodes = run_cfg.total_steps // (buffer_cfg.size * buffer_cfg.num_envs)
    total_steps = 0

    for episode in range(num_episodes):
        metrics = trainer.train_episode()
        total_steps += buffer_cfg.size * buffer_cfg.num_envs

        if episode % run_cfg.log_every == 0:
            print(f"Episode {episode}/{num_episodes} | Steps: {total_steps}/{run_cfg.total_steps}")
            print(f"Average reward: {metrics.mean_reward:.2f}")
            print(f"Policy loss: {metrics.policy_loss:.3f}")
            print(f"Value loss: {metrics.value_loss:.3f}")
            print(f"Last gradient norm: {metrics.last_grad_norm:.3f}")
            print("-" * 50)


if __name__ == "__main__":
    main()
