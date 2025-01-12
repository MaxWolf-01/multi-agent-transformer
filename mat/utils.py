import numpy as np
from jaxtyping import Float
from torch import nn


def init_weights(m: nn.Module, gain: float = 0.01, use_relu_gain: bool = False):
    if isinstance(m, nn.Linear):
        if use_relu_gain:
            gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def get_gae_returns_and_advantages(
    rewards: Float[np.ndarray, "steps envs agents 1"],
    values: Float[np.ndarray, "steps envs agents 1"],
    next_value: Float[np.ndarray, "envs agents 1"],
    gamma: float,
    gae_lambda: float,
) -> tuple[Float[np.ndarray, "steps envs agents 1"], Float[np.ndarray, "steps envs agents 1"]]:
    """Compute returns and advantages using Generalized Advantage Estimation (GAE)"""
    steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_advantage = 0

    for t in reversed(range(steps)):
        if t == steps - 1:
            next_value_t = next_value
        else:
            next_value_t = values[t + 1]
        # TD error = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value_t - values[t]
        # GAE = sum (γλ)^k δ_{t+k} | Computed recursively: A_t = δ_t + γλA_{t+1}
        advantages[t] = delta + gamma * gae_lambda * last_advantage
        last_advantage = advantages[t]
    returns = advantages + values
    return returns, advantages
