from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from mat.utils import get_gae_returns_and_advantages


@dataclass
class BufferConfig:
    size: int
    num_envs: int
    num_agents: int
    obs_shape: tuple[int, ...]
    action_shape: tuple[int, ...]


class BufferSample(NamedTuple):
    """Data sampled from buffer for PPO updates.
    All tensors have shape (batch_size, num_agents, feature_dims) where batch_size = num_envs * rollout_steps.
    """

    obs: Float[Tensor, "batch agents *obs"]
    actions: Float[Tensor, "batch agents *action"]
    old_values: Float[Tensor, "batch agents 1"]
    old_action_log_probs: Float[Tensor, "batch agents action"]
    advantages: Float[Tensor, "batch agents 1"]
    returns: Float[Tensor, "batch agents 1"]
    active_masks: Float[Tensor, "batch agents 1"] | None


class Buffer:
    """Stores trajectories collected from multiple parallel environments.

    - Uses a circular buffer with size+1 slots for observations/values because we need the final state's value estimate for bootstrapping returns
    - self.step tracks current position in circular buffer (0 to size-1)
    - self.full indicates whether we've collected a complete trajectory worth of data
    """

    def __init__(self, config: BufferConfig):
        self.cfg = config
        cfg, size, num_envs, num_agents = config, config.size, config.num_envs, config.num_agents

        # add extra step (+1) to observations/values/dones for bootstrapping returns computation
        self.obs = np.zeros((size + 1, num_envs, num_agents, *cfg.obs_shape), dtype=np.float32)
        self.values = np.zeros((size + 1, num_envs, num_agents, 1), dtype=np.float32)
        self.dones = np.zeros((size + 1, num_envs, num_agents, 1), dtype=np.float32)
        # regular trajectory data
        self.actions = np.zeros((size, num_envs, num_agents, *cfg.action_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((size, num_envs, num_agents, *cfg.action_shape), dtype=np.float32)
        self.rewards = np.zeros((size, num_envs, num_agents, 1), dtype=np.float32)
        # optional mask for inactive agents (e.g., dead agents in some environments)
        self.active_masks = np.ones((size + 1, num_envs, num_agents, 1), dtype=np.float32)
        # computed after collecting complete trajectory
        self.returns = None
        self.advantages = None

        self.full = False
        self.step = 0

    def insert(
        self,
        obs: Float[np.ndarray, "envs agents *obs_shape"],
        actions: Float[np.ndarray, "envs agents *action"],
        action_log_probs: Float[np.ndarray, "envs agents *action"],
        values: Float[np.ndarray, "envs agents 1"],
        rewards: Float[np.ndarray, "envs agents 1"],
        dones: Float[np.ndarray, "envs agents 1"],
        active_masks: Float[np.ndarray, "envs agents 1"] | None = None,
    ) -> None:
        """Insert a new transition into the buffer.

        The +1 offset for obs/dones/masks aligns the data so that for calling this method at step t:
        - obs[t+1] is the observation after taking action[t]
        - action[t] is the action taken in state obs[t]
        - reward[t] is the reward received after action[t]
        - done[t+1] indicates if obs[t+1] was terminal
        - values[t] is the estimated value of obs[t]
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

    def compute_returns_and_advantages(
        self,
        next_value: Float[np.ndarray, "envs agents 1"],
        gamma: float,
        gae_lambda: float,
        normalize_advantage: bool = True,
    ) -> None:
        self.returns, self.advantages = get_gae_returns_and_advantages(
            rewards=self.rewards,
            values=self.values[:-1],  # remove last value used for bootstrap (because it requires next_value)
            next_value=next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_minibatches(self, num_minibatches: int, device: torch.device) -> BufferSample:
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

            def _cast(x: np.ndarray) -> torch.Tensor:
                return torch.from_numpy(x.reshape(-1, *x.shape[2:])[mb_inds]).to(device)

            yield BufferSample(
                obs=_cast(self.obs[:-1]),
                actions=_cast(self.actions),
                old_values=_cast(self.values[:-1]),
                old_action_log_probs=_cast(self.action_log_probs),
                advantages=_cast(self.advantages),
                returns=_cast(self.returns),
                active_masks=_cast(self.active_masks[:-1]) if self.active_masks is not None else None,
            )

    def after_update(self) -> None:
        """Last timestep becomes starting point for the next. Called after policy update."""
        self.obs[0] = self.obs[-1]
        self.dones[0] = self.dones[-1]
        self.active_masks[0] = self.active_masks[-1]

        self.returns = None
        self.advantages = None
