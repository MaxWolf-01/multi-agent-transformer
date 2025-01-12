from typing import Any

import numpy as np
import torch
from pettingzoo.mpe import simple_tag_v3, simple_world_comm_v3

from mat.buffer import Buffer
from mat.mat import MAT
from mat.runners.base import EnvRunner


class MPERunner(EnvRunner):
    """Runner for Multi-Particle environments from PettingZoo."""

    SUPPORTED_ENVS = {
        "simple_tag_v3": simple_tag_v3.parallel_env,
        "simple_world_comm_v3": simple_world_comm_v3.parallel_env,
        # TODO add the rest
    }

    @staticmethod
    def get_env(env_id: str, env_kwargs: dict):
        if env_id not in MPERunner.SUPPORTED_ENVS:
            raise ValueError(f"Unknown MPE environment: {env_id}")
        return MPERunner.SUPPORTED_ENVS[env_id](**env_kwargs)

    def __init__(self, env, policy: MAT, buffer: Buffer):
        super().__init__(buffer)
        self.env = env
        self.policy = policy
        self.buffer = buffer

        self.agent_ids = self.env.agents  # for consistent ordering (MPE envs have fixed agents)
        self.num_agents = len(self.agent_ids)

    def collect_rollout(self) -> torch.Tensor:
        """Collect a rollout using the current policy. Returns value estimate for final state (for bootstrapping)."""
        observations, _ = self.env.reset()
        for step in range(self.buffer.cfg.size):
            obs_tensor = self._dict_to_tensor(observations)

            with torch.no_grad():
                policy_output = self.policy.get_actions(obs=obs_tensor)

            actions = {agent_id: action.cpu().numpy() for agent_id, action in zip(self.agent_ids, policy_output.actions)}
            observations, rewards, terminations, truncations, infos = self.env.step(actions)

            self.buffer.insert(
                obs=obs_tensor.numpy(),
                actions=policy_output.actions.numpy(),
                action_log_probs=policy_output.action_log_probs.numpy(),
                values=policy_output.values.numpy(),
                rewards=self._dict_to_array(rewards),
                dones=self._dict_to_array(terminations) | self._dict_to_array(truncations),
                active_masks=None,
            )

        # get value estimate for final state
        obs_tensor = self._dict_to_tensor(observations)
        with torch.no_grad():
            next_values = self.policy.get_values(obs_tensor)

        return next_values

    def _dict_to_tensor(self, obs_dict: dict[str, np.ndarray]) -> torch.Tensor:
        """Convert dict of observations to tensor of shape (num_agents, *obs_shape)."""
        return torch.tensor(np.stack([obs_dict[agent_id] for agent_id in self.agent_ids]), dtype=torch.float32)

    def _dict_to_array(self, val_dict: dict[str, Any]) -> np.ndarray:
        """Convert dict of values to array of shape (num_agents, 1)."""
        return np.expand_dims(np.array([val_dict[agent_id] for agent_id in self.agent_ids]), axis=-1)
