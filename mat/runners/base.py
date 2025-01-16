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
        """Run policy in environment and collect transitions. Return value estimate for final state."""

        # obs = self.env.reset()
        # next_obs = None
        # for step in range(self.buffer.cfg.size):
        #     with torch.no_grad():
        #         policy_output = self.policy.get_actions(obs=obs)
        #     next_obs, rewards, terminations, truncations, info = self.env.step(policy_output.actions)
        #     self.buffer.insert(
        #         obs=obs,
        #         actions=policy_output.actions,
        #         action_log_probs=policy_output.action_log_probs,
        #         values=policy_output.values.numpy(),
        #         rewards=rewards,
        #         dones=terminations | truncations,
        #         active_masks=None,
        #     )
        #
        #     obs = next_obs
        #
        # with torch.no_grad():
        #     next_values = self.policy.get_values(torch.tensor(next_obs))
        # return next_values
