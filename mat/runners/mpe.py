from dataclasses import dataclass

import numpy as np
import supersuit as ss
import torch
from jaxtyping import Float
from pettingzoo.mpe import simple_spread_v3
from torch import Tensor

from mat.buffer import Buffer
from mat.mat import MAT
from mat.runners.base import EnvRunner, RunnerConfig


@dataclass
class MPERunnerConfig(RunnerConfig):
    env_id: str


class MPERunner(EnvRunner):
    cfg: MPERunnerConfig
    SUPPORTED_ENVS = {
        "simple_spread_v3": simple_spread_v3.parallel_env,
    }

    @staticmethod
    def get_env(env_id: str, env_kwargs: dict):
        if env_id not in MPERunner.SUPPORTED_ENVS:
            raise ValueError(f"Unknown MPE environment: {env_id}")
        return MPERunner.SUPPORTED_ENVS[env_id](**env_kwargs)

    def __init__(self, config: MPERunnerConfig, buffer: Buffer, policy: MAT):
        """If render_kwargs is given, a separate env for rendering will be created."""
        super().__init__(config, buffer, policy)
        env = self.get_env(config.env_id, config.env_kwargs)
        self.env = ss.concat_vec_envs_v1(ss.pettingzoo_env_to_vec_env_v1(env, array_act=False), num_vec_envs=config.num_envs)

        env.reset()  # required to access agents / num_agents attribute
        self.num_agents = env.num_agents  # not equal to self.env.agents, which is num_paralell_envs * num_agents

        self.next_obs = self._get_obs(self.env.reset()[0])
        self.render_env = (
            self.get_env(env_id=config.env_id, env_kwargs=config.env_kwargs | dict(render_mode="human"))
            if config.render
            else None
        )
        self.render_obs, _ = self.render_env.reset() if config.render else (None, None)

    @torch.inference_mode()
    def collect_rollout(self) -> Float[Tensor, "b agents"]:
        """Collects a rollout using the current policy and returns the value estimate for final state."""
        self._permute_agents()
        for step in range(self.buffer.cfg.length):
            policy_output = self.policy.get_actions(obs=(obs := self.next_obs))
            actions = policy_output.actions.cpu().numpy()[:, self.inverse_agent_perm]  # (batch, agents)
            self.next_obs, rewards, terminations, truncations, infos = self.env.step(actions.reshape(-1))
            self.next_obs = self._get_obs(self.next_obs)
            self.buffer.insert(
                obs=obs.cpu().numpy(),
                actions=actions[:, :, None],
                action_log_probs=policy_output.action_log_probs.cpu().numpy(),
                values=policy_output.values.cpu().numpy(),
                rewards=rewards.reshape(self.cfg.num_envs, self.num_agents),
                dones=(terminations | truncations).reshape(self.cfg.num_envs, self.num_agents),
                active_masks=None,
            )
            if self.render_env is not None:
                self._render_step()
        return self.policy.get_values(self.next_obs)

    def _render_step(self) -> None:
        obs = np.array([self.render_obs[agent] for agent in self.render_env.agents]).reshape(1, self.num_agents, -1)
        policy_output = self.policy.get_actions(obs=torch.tensor(obs, device=self.device))
        actions = {agent: action for agent, action in zip(self.render_env.agents, policy_output.actions[0].cpu().numpy())}
        self.render_obs, _, terminations, truncations, _ = self.render_env.step(actions)
        if any(terminations.values()) or any(truncations.values()):  # pettingzoo doesn't automatically reset :/
            self.render_obs, _ = self.render_env.reset()
