import numpy as np
import supersuit as ss
import torch
from jaxtyping import Float
from pettingzoo.mpe import simple_spread_v3
from torch import Tensor

from mat.buffer import Buffer
from mat.mat import MAT
from mat.runners.base import EnvRunner


class MPERunner(EnvRunner):
    SUPPORTED_ENVS = {
        "simple_spread_v3": simple_spread_v3.parallel_env,
    }

    @staticmethod
    def get_env(env_id: str, env_kwargs: dict):
        if env_id not in MPERunner.SUPPORTED_ENVS:
            raise ValueError(f"Unknown MPE environment: {env_id}")
        return MPERunner.SUPPORTED_ENVS[env_id](**env_kwargs)

    def __init__(self, env, policy: MAT, buffer: Buffer, num_envs: int, device: str | torch.device, render_kwargs: dict | None):
        """If render_kwargs is given, a separate env for rendering will be created."""
        super().__init__(device, buffer)
        self.env = ss.concat_vec_envs_v1(ss.pettingzoo_env_to_vec_env_v1(env), num_vec_envs=num_envs)
        self.render_env = self.get_env(env_id=env.metadata["name"], env_kwargs=render_kwargs) if render_kwargs else None
        self.policy = policy
        self.buffer = buffer
        self.num_parallel_envs = num_envs
        self.device = device

        self.num_agents = len(env.agents)  # not equal to self.env.agents, which is num_paralell_envs * num_agents
        self.act_dim = env.action_space(env.agents[0]).n

    @torch.inference_mode()
    def collect_rollout(self) -> Float[Tensor, "b agents"]:
        """Collect a rollout using the current policy. Returns value estimate for final state (for bootstrapping)."""
        observations, _ = self.env.reset()
        render_obs = self.render_env.reset()[0] if self.render_env is not None else None
        for step in range(self.buffer.cfg.size):
            obs_reshaped = observations.reshape(self.num_parallel_envs, self.num_agents, -1)
            obs_tensor = torch.as_tensor(obs_reshaped, device=self.device).float()

            policy_output = self.policy.get_actions(obs=obs_tensor)
            actions = policy_output.actions.cpu().numpy()  # (batch, agents)
            observations, rewards, terminations, truncations, infos = self.env.step(actions.reshape(-1))

            rewards = rewards.reshape(self.num_parallel_envs, self.num_agents, 1)
            dones = (terminations | truncations).reshape(self.num_parallel_envs, self.num_agents)

            self.buffer.insert(
                obs=obs_tensor.cpu().numpy(),
                actions=actions[:, :, None],
                action_log_probs=policy_output.action_log_probs.cpu().numpy(),
                values=policy_output.values.squeeze(-1).cpu().numpy(),
                rewards=rewards.squeeze(-1),
                dones=dones,
                active_masks=None,
            )
            if render_obs is not None:
                render_obs = self._render_step(render_obs)

        # get value estimate for final state
        obs_reshaped = torch.tensor(observations.reshape(self.num_parallel_envs, self.num_agents, -1), device=self.device)
        next_values = self.policy.get_values(obs_reshaped).squeeze(-1)
        return next_values

    def _render_step(self, render_obs: dict) -> dict:
        render_policy_output = self.policy.get_actions(
            obs=torch.as_tensor(
                np.array([render_obs[agent] for agent in self.render_env.agents]).reshape(1, self.num_agents, -1),
                device=self.device,
            ).float()
        )
        return self.render_env.step(
            {agent: action for agent, action in zip(self.render_env.agents, render_policy_output.actions[0].cpu().numpy())}
        )[0]
