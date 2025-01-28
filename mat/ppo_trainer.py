from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from mat.mat import MAT, MATTrainingOutput
from mat.runners.base import EnvRunner


@dataclass
class TrainerConfig:
    lr: float
    eps: float
    weight_decay: float
    max_grad_norm: float

    # PPO
    num_ppo_epochs: int
    minibatch_size: int
    clip_param: float
    value_loss_coef: float
    entropy_coef: float
    gamma: float
    gae_lambda: float
    use_clipped_value_loss: bool
    normalize_advantage: bool  # batch-level normalization
    use_huber_loss: bool
    huber_delta: float

    device: str | torch.device


@dataclass
class Metrics:
    loss: float
    policy_objective: float
    value_loss: float
    entropy: float
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
        self.loss_fn = (
            partial(F.huber_loss, reduction="none", delta=config.huber_delta)
            if self.cfg.use_huber_loss
            else partial(F.mse_loss, reduction="none")
        )
        self._clip_range = (1 - self.cfg.clip_param, 1 + self.cfg.clip_param)
        self._clear_metrics = lambda: Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._metrics = self._clear_metrics()

    def _compute_policy_objective(
        self,
        action_log_probs: Float[Tensor, "batch agents action"],
        old_action_log_probs: Float[Tensor, "batch agents action"],
        advantages: Float[Tensor, "batch agents 1"],
        active_masks: Float[Tensor, "batch agents 1"] | None = None,
    ) -> Float[Tensor, "1"]:
        ratio = torch.exp(action_log_probs - old_action_log_probs)  # log(a/b)=log(a)-log(b)
        objective = torch.min(ratio * advantages, ratio.clamp(*self._clip_range) * advantages)
        return objective.mean() if active_masks is None else (objective * active_masks).sum() / active_masks.sum()

    def _compute_value_loss(
        self,
        values: Float[Tensor, "batch agents"],
        returns: Float[Tensor, "batch agents"],
        old_values: Float[Tensor, "batch agents"],
        active_masks: Float[Tensor, "batch agents"] | None = None,
    ) -> Float[Tensor, "1"]:
        loss = self.loss_fn(values, returns)
        if self.cfg.use_clipped_value_loss:
            clipped_vals = values.clamp(old_values - self.cfg.clip_param, old_values + self.cfg.clip_param)
            loss = torch.max(loss, self.loss_fn(clipped_vals, returns))
        return loss.mean() if active_masks is None else (loss * active_masks).sum() / active_masks.sum()

    def _update_policy(self) -> None:
        """Update policy using PPO on the collected rollout in the buffer."""
        for _ in range(self.cfg.num_ppo_epochs):
            for batch in self.runner.buffer.get_minibatches(mb_size=self.cfg.minibatch_size, device=self.cfg.device):
                policy_out: MATTrainingOutput = self.policy(obs=batch.obs, actions=batch.actions)
                policy_objective = self._compute_policy_objective(
                    action_log_probs=policy_out.action_log_probs
                    if len(policy_out.action_log_probs.shape) > 2
                    else policy_out.action_log_probs.unsqueeze(-1),
                    old_action_log_probs=batch.old_action_log_probs,
                    advantages=batch.advantages.unsqueeze(-1),
                    active_masks=batch.active_masks,
                )
                value_loss = self._compute_value_loss(
                    values=policy_out.values,
                    returns=batch.returns,
                    old_values=batch.old_values,
                    active_masks=batch.active_masks,
                )
                entropy = (
                    policy_out.entropy.mean()
                    if batch.active_masks is None
                    else (policy_out.entropy * batch.active_masks).sum() / batch.active_masks.sum()
                )
                loss = -(policy_objective - value_loss * self.cfg.value_loss_coef + entropy * self.cfg.entropy_coef)
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                self._metrics.loss += loss.item()
                self._metrics.policy_objective += policy_objective.item()
                self._metrics.value_loss += value_loss.item()
                self._metrics.entropy += entropy.item()
                self._metrics.last_grad_norm = grad_norm.item()
        num_updates = self.cfg.num_ppo_epochs * (self.runner.buffer.batch_size // self.cfg.minibatch_size)
        self._metrics.loss /= num_updates
        self._metrics.policy_objective /= num_updates
        self._metrics.value_loss /= num_updates
        self._metrics.entropy /= num_updates

    def train_iteration(self) -> Metrics:
        """
        1. Collect experience using runner
        2. Compute advantages and returns
        3. Update policy multiple times with PPO
        4. Return metrics
        """
        self._metrics = self._clear_metrics()
        last_value = self.runner.collect_rollout()
        self.runner.buffer.compute_returns_and_advantages(
            last_value=last_value.cpu().numpy(),
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            normalize_advantage=self.cfg.normalize_advantage,
        )
        self._update_policy()
        self.runner.buffer.after_update()  # reset buffer (keeps last observation for next iteration)
        # TODO lr decay
        self._metrics.mean_reward = float(self.runner.buffer.rewards.mean())
        return self._metrics
