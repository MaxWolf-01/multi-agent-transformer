import abc
from dataclasses import dataclass
from typing import Any

import einops
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from mat.buffer import Buffer
from mat.mat import MAT


@dataclass
class RunnerConfig:
    env_kwargs: dict[str, Any]
    num_agents: int
    num_envs: int  # number of parallel environments
    use_agent_id_enc: bool  # whether to add one-hot agent id to obs
    permute_agents: bool  # whether to permute input observation and output actions
    device: str | torch.device
    render: bool


class EnvRunner(abc.ABC):
    """Handles environment interaction and data collection."""

    def __init__(self, config: RunnerConfig, buffer: Buffer, policy: MAT):
        self.cfg = config
        self.buffer = buffer
        self.policy = policy
        self.num_agents = config.num_agents
        self.agent_perm = self.inverse_agent_perm = torch.arange(self.num_agents)

    @abc.abstractmethod
    def collect_rollout(self) -> Float[Tensor, "b agents"]:
        """Run policy in environment and collect transitions. Return value estimate for final state.
        Note: Do not reset the env every time this is called. Gymnasium envs (usually...) automatically reset when done.
        ```pseudocode example
        self._permute_agents() # TODO: permute per rollout or per step?
        for step in range(self.buffer.cfg.size):
            with torch.no_grad():
                policy_output = self.policy.get_actions(obs=self._get_obs(self.obs))
            next_obs, rewards, terminations, truncations, info = self.env.step(policy_output.actions[:, self.agent_perm])
            self.buffer.insert(self.obs, policy_output.actions, ...)
            self.obs = next_obs
        with torch.no_grad():
            next_values = self.policy.get_values(torch.tensor(next_obs))
        return next_values
        ```
        """

    def _get_obs(self, o: Float[np.ndarray, "(envs agents) obs"], num_envs: int | None = None) -> Float[Tensor, "b agents obs"]:
        """Convert observation to tensor and permute agents if necessary, optionally adding one-hot agent id."""
        num_envs = num_envs or self.cfg.num_envs
        o = torch.tensor(o, device=self.cfg.device, dtype=torch.float32).view(num_envs, self.num_agents, -1)
        o = o[:, self.agent_perm]
        if not self.cfg.use_agent_id_enc:
            return o
        agent_ids = F.one_hot(torch.arange(self.num_agents, device=self.cfg.device), num_classes=self.num_agents)
        agent_ids = agent_ids[self.agent_perm, :]
        agent_ids = einops.repeat(agent_ids, "agents id -> b agents id", b=num_envs)
        return torch.cat([o, agent_ids], dim=-1)

    def _permute_agents(self) -> None:
        """Randomly permute the agent order. Called at the beginning of each rollout."""
        if self.cfg.permute_agents:
            self.agent_perm = np.random.permutation(self.num_agents)
            self.inverse_agent_perm = np.argsort(self.agent_perm)
