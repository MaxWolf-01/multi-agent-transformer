from dataclasses import dataclass
from typing import Iterator, NamedTuple

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from mat.utils import get_gae_returns_and_advantages


class Trajectory(NamedTuple):
    obs: Float[Tensor, "batch agents *obs_shape"]  # batch = trajecotry length, agents = number of agents
    actions: Float[Tensor, "batch agents act"]  # act = scalar index if discrete, vector if continuous
    old_values: Float[Tensor, "batch agents"]
    old_action_log_probs: Float[Tensor, "batch agents act"]
    advantages: Float[Tensor, "batch agents"]
    returns: Float[Tensor, "batch agents"]
    active_masks: Float[Tensor, "batch agents"] | None


@dataclass
class BufferConfig:
    size: int
    num_envs: int
    num_agents: int
    obs_shape: tuple[int, ...]
    action_dim: int
    active_masks: bool = False


class Buffer:
    """Stores trajectories collected from multiple parallel environments.
    - Uses a circular buffer with size+1 slots for observations/values because we need the final state's value estimate for bootstrapping returns
    - self.step tracks current position in circular buffer (0 to size-1)
    - self.full indicates whether we've collected a complete trajectory worth of data
    """

    def __init__(self, config: BufferConfig):
        self.cfg = config
        cfg, size, num_envs, num_agents = config, config.size, config.num_envs, config.num_agents
        self.values = np.zeros((size, num_envs, num_agents), dtype=np.float32)
        self.actions = np.zeros((size, num_envs, num_agents, cfg.action_dim), dtype=np.float32)
        self.action_log_probs = np.zeros((size, num_envs, num_agents), dtype=np.float32)
        self.rewards = np.zeros((size, num_envs, num_agents), dtype=np.float32)
        # +1 to observations/dones/masks for last observation -> start of next trajectory; last val is passed separately
        self.obs = np.zeros((size + 1, num_envs, num_agents, *cfg.obs_shape), dtype=np.float32)
        self.dones = np.zeros((size + 1, num_envs, num_agents), dtype=np.float32)
        # optional mask for inactive agents (e.g., dead agents in some environments)
        self.active_masks = np.ones((size + 1, num_envs, num_agents), dtype=np.float32) if cfg.active_masks else None
        # computed after collecting complete trajectory
        self.returns = None
        self.advantages = None

        self.full = False
        self.step = 0

    def insert(
        self,
        obs: Float[np.ndarray, "envs agents *obs_shape"],
        actions: Float[np.ndarray, "envs agents act"],
        action_log_probs: Float[np.ndarray, "envs agents act"],
        values: Float[np.ndarray, "envs agents"],
        rewards: Float[np.ndarray, "envs agents"],
        dones: Float[np.ndarray, "envs agents"],
        active_masks: Float[np.ndarray, "envs agents"] | None = None,
    ) -> None:
        """Insert a new transition into the buffer.

        Uses +1 offset for state data (obs, dones, active_masks) so that at step t:
        - obs[t+1], done[t+1] represent the state after taking action[t]
        - action[t], reward[t], value[t] represent what happened in state obs[t]
        """
        self.obs[self.step + 1] = obs
        self.actions[self.step] = actions
        self.action_log_probs[self.step] = action_log_probs
        self.values[self.step] = values
        self.rewards[self.step] = rewards
        self.dones[self.step + 1] = dones
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks

        self.step = (self.step + 1) % self.cfg.size
        if self.step == 0:
            self.full = True

    def get_minibatches(self, num_minibatches: int, device: torch.device) -> Iterator[Trajectory]:
        """
        Flattens the env and time dimensions to create a batch of transitions (batch_size, *dims) , then splits this batch into minibatches.
        We remove the final observation/value since they were only needed for bootstrapping returns computation.
        """
        num_envs, size = self.cfg.num_envs, self.cfg.size
        batch_size = num_envs * size
        mini_batch_size = batch_size // num_minibatches

        indices = np.random.permutation(batch_size)
        for start_idx in range(0, batch_size, mini_batch_size):
            end_idx = start_idx + mini_batch_size
            mb_inds = indices[start_idx:end_idx]

            def _get_minibatch(x: np.ndarray) -> torch.Tensor:  # (size, num_envs, num_agents, *) -> (bs, num_agents, *)
                return torch.as_tensor(x.reshape(-1, *x.shape[2:])[mb_inds], device=device)

            yield Trajectory(
                obs=_get_minibatch(self.obs[:-1]),
                actions=_get_minibatch(self.actions),
                old_values=_get_minibatch(self.values),
                old_action_log_probs=_get_minibatch(self.action_log_probs),
                advantages=_get_minibatch(self.advantages),
                returns=_get_minibatch(self.returns),
                active_masks=_get_minibatch(self.active_masks[:-1]) if self.active_masks is not None else None,
            )

    def compute_returns_and_advantages(
        self,
        last_value: Float[np.ndarray, "envs agents"],
        gamma: float,
        gae_lambda: float,
        normalize_advantage: bool,
    ) -> None:
        self.returns, self.advantages = get_gae_returns_and_advantages(
            rewards=self.rewards,
            values=np.concatenate([self.values, last_value[None]]),
            dones=self.dones[:-1],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        if normalize_advantage:
            advantages = self.advantages * self.active_masks[:-1] if self.active_masks is not None else self.advantages
            self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def after_update(self) -> None:
        """Last timestep becomes starting point for the next. Called after policy update."""
        self.obs[0] = self.obs[-1]
        self.dones[0] = self.dones[-1]
        if self.active_masks is not None:
            self.active_masks[0] = self.active_masks[-1]

        self.returns = None
        self.advantages = None
