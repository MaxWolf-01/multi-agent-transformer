from dataclasses import dataclass

import numpy as np
import supersuit as ss
import torch
from gymnasium_robotics.envs.multiagent_mujoco.mamujoco_v1 import parallel_env
from jaxtyping import Float
from torch import Tensor

from mat.buffer import Buffer
from mat.mat import MAT
from mat.runners.base import EnvRunner, RunnerConfig


@dataclass
class MujocoRunnerConfig(RunnerConfig): ...


class MujocoRunner(EnvRunner):
    def __init__(self, config: MujocoRunnerConfig, buffer: Buffer, policy: MAT):
        super().__init__(config, buffer, policy)
        base_env = parallel_env(**config.env_kwargs | dict(render_mode=None))  # cant & dont want to render parallel
        base_env.unwrapped.render_mode = None  # gym api is weird
        obs, _ = base_env.reset()  # required to access num_agents
        self.obs_dim = len(base_env.map_local_observations_to_global_state(obs))
        self.act_dim = max([space.shape[0] for space in base_env.action_spaces.values()])
        self.num_agents = base_env.num_agents

        self.env = ss.pettingzoo_env_to_vec_env_v1(base_env, array_act=True)
        self.env = ss.concat_vec_envs_v1(self.env, num_vec_envs=config.num_envs)

        self.next_obs = self._get_obs(self.env.reset()[0])
        self.render_env = parallel_env(**config.env_kwargs | dict(render_mode="human")) if config.render else None
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
                actions=actions,
                action_log_probs=policy_output.action_log_probs.squeeze().cpu().numpy(),
                values=policy_output.values.cpu().numpy(),
                rewards=rewards.reshape(self.cfg.num_envs, self.num_agents),
                dones=(terminations | truncations).reshape(self.cfg.num_envs, self.num_agents),
                active_masks=None,
            )

            if self.render_env is not None:
                self._render_step()

        return self.policy.get_values(self.next_obs)

    def _render_step(self) -> None:
        obs = self._get_obs(np.array([self.render_obs[agent] for agent in self.render_env.agents]), num_envs=1)
        policy_output = self.policy.get_actions(obs=obs)
        actions = {agent: action for agent, action in zip(self.render_env.agents, policy_output.actions[0].cpu().numpy())}
        self.render_obs, _, terminations, truncations, _ = self.render_env.step(actions)
        if any(terminations.values()) or any(truncations.values()):
            self.render_obs, _ = self.render_env.reset()
