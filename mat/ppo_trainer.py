from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from mat.mat import MAT
from mat.runners.base import EnvRunner


# TODO cleanup ifs in loss computations


@dataclass
class TrainerConfig:
    """Configuration for the PPO trainer."""

    lr: float
    eps: float
    weight_decay: float
    max_grad_norm: float

    num_episodes: int

    # PPO
    num_minibatches: int
    clip_param: float
    value_loss_coef: float
    entropy_coef: float
    gamma: float
    gae_lambda: float
    use_clipped_value_loss: bool
    normalize_advantage: bool
    use_huber_loss: bool
    huber_delta: float

    device: str | torch.device


@dataclass
class Metrics:
    """Metrics from a training episode."""

    policy_loss: float
    value_loss: float
    last_grad_norm: float
    mean_reward: float


class PPOTrainer:
    """Handles training loop and policy updates using PPO."""

    def __init__(self, config: TrainerConfig, policy: MAT, runner: EnvRunner):
        self.cfg = config
        self.policy = policy
        self.runner = runner
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.lr, eps=config.eps, weight_decay=config.weight_decay
        )
        self._metrics = None

    @property
    def metrics(self) -> Metrics:
        if self._metrics is None:
            raise RuntimeError("No metrics available yet. Run train_episode() first.")
        return self._metrics

    def _compute_policy_loss(
        self,
        action_log_probs: Float[Tensor, "batch agents action"],
        old_action_log_probs: Float[Tensor, "batch agents action"],
        advantages: Float[Tensor, "batch agents 1"],
        active_masks: Float[Tensor, "batch agents 1"] | None = None,
    ) -> Float[Tensor, "1"]:
        """Compute PPO policy loss with clipping."""
        ratio = torch.exp(action_log_probs - old_action_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param) * advantages

        policy_loss = -torch.min(surr1, surr2)

        if active_masks is not None:
            policy_loss = (policy_loss * active_masks).sum() / active_masks.sum()
        else:
            policy_loss = policy_loss.mean()

        return policy_loss

    def _compute_value_loss(
        self,
        values: Float[Tensor, "batch agents"],
        returns: Float[Tensor, "batch agents"],
        old_values: Float[Tensor, "batch agents"] | None = None,
        active_masks: Float[Tensor, "batch agents"] | None = None,
    ) -> Float[Tensor, "1"]:
        """Compute value function loss, optionally with clipping."""
        base_loss_fn = F.huber_loss if self.cfg.use_huber_loss else F.mse_loss

        if self.cfg.use_huber_loss:
            value_loss = base_loss_fn(values, returns, reduction="none", delta=self.cfg.huber_delta)
        else:
            value_loss = base_loss_fn(values, returns, reduction="none")

        if self.cfg.use_clipped_value_loss and old_values is not None:
            value_pred_clipped = old_values + (values - old_values).clamp(-self.cfg.clip_param, self.cfg.clip_param)
            if self.cfg.use_huber_loss:
                value_loss_clipped = base_loss_fn(value_pred_clipped, returns, reduction="none", delta=self.cfg.huber_delta)
            else:
                value_loss_clipped = base_loss_fn(value_pred_clipped, returns, reduction="none")
            value_loss = torch.max(value_loss, value_loss_clipped)

        if active_masks is not None:
            value_loss = (value_loss * active_masks).sum() / active_masks.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss * self.cfg.value_loss_coef

    def _update_policy(self) -> dict[str, Any]:
        """Update policy using PPO on the collected rollout in the buffer."""
        metrics = {
            "mean_policy_loss": 0.0,
            "mean_value_loss": 0.0,
            "last_grad_norm": 0.0,
        }
        for _ in range(self.cfg.num_episodes):
            for batch in self.runner.buffer.get_minibatches(num_minibatches=self.cfg.num_minibatches, device=self.cfg.device):
                policy_output = self.policy(obs=batch.obs, actions=batch.actions)
                policy_loss = self._compute_policy_loss(
                    action_log_probs=policy_output.action_log_probs,
                    old_action_log_probs=batch.old_action_log_probs,
                    advantages=batch.advantages,
                    active_masks=batch.active_masks,
                )
                value_loss = self._compute_value_loss(
                    values=policy_output.values,
                    returns=batch.returns,
                    old_values=batch.old_values if self.cfg.use_clipped_value_loss else None,
                    active_masks=batch.active_masks,
                )
                loss = policy_loss + value_loss
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.cfg.max_grad_norm,
                )
                self.optimizer.step()

                metrics["mean_policy_loss"] += policy_loss.item()
                metrics["mean_value_loss"] += value_loss.item()
                metrics["last_grad_norm"] = grad_norm.item()
        num_updates = self.cfg.num_episodes * self.cfg.num_minibatches
        metrics["mean_policy_loss"] /= num_updates
        metrics["mean_value_loss"] /= num_updates
        return metrics

    def train_episode(self) -> Metrics:
        """Run one training episode.

        1. Collect experience using runner
        2. Compute advantages and returns
        3. Update policy multiple times with PPO
        4. Return metrics
        """
        next_values = self.runner.collect_rollout()

        self.runner.buffer.compute_returns_and_advantages(
            next_value=next_values.cpu().numpy(),
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            normalize_advantage=self.cfg.normalize_advantage,
        )
        episode_metrics = self._update_policy()
        self.runner.buffer.after_update()  # reset buffer (keeps last observation for next episode)

        mean_reward = float(self.runner.buffer.rewards.mean())
        self._metrics = Metrics(
            policy_loss=episode_metrics["mean_policy_loss"],
            value_loss=episode_metrics["mean_value_loss"],
            last_grad_norm=episode_metrics["last_grad_norm"],
            mean_reward=mean_reward,
        )
        return self._metrics
