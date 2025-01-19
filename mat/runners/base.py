import abc

import torch
from jaxtyping import Float
from torch import Tensor

from mat.buffer import Buffer


class EnvRunner(abc.ABC):
    """Handles environment interaction and data collection."""

    def __init__(self, device: str | torch.device, buffer: Buffer):
        self.device = device
        self.buffer = buffer

    @abc.abstractmethod
    def collect_rollout(self) -> Float[Tensor, "b agents"]:
        """Run policy in environment and collect transitions. Return value estimate for final state.
        Note: Do not reset the env every time this is called. Gymnasium envs (usually...) automatically reset when done.
        ```
        for step in range(self.buffer.cfg.size):
            with torch.no_grad():
                policy_output = self.policy.get_actions(obs=self.obs)
            next_obs, rewards, terminations, truncations, info = self.env.step(policy_output.actions)
            self.buffer.insert(...)
            self.obs = next_obs
        with torch.no_grad():
            next_values = self.policy.get_values(torch.tensor(next_obs))
        return next_values
        ```
        """
