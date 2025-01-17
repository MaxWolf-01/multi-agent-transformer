import abc
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.distributions import Categorical, Normal

from mat.decoder import DecentralizedMlpDecoder, TransformerDecoder


@dataclass
class SamplerConfig:
    num_agents: int
    act_dim: int
    device: str | torch.device
    dtype: torch.dtype


class Sampler(abc.ABC):
    def __init__(self, config: SamplerConfig):
        self.cfg = config
        self.tprops = dict(device=config.device, dtype=config.dtype)


class ParallelDiscreteSample(NamedTuple):
    actions: Float[Tensor, "b agents"]
    entropy: Float[Tensor, "b"]


class AutoregressiveDiscreteSample(NamedTuple):
    actions: Float[Tensor, "b agents"]
    log_probs: Float[Tensor, "b agents"]


@dataclass
class DiscreteSamplerConfig(SamplerConfig):
    start_token: int = 1


class DiscreteSampler(Sampler):
    cfg: DiscreteSamplerConfig

    def __init__(self, config: DiscreteSamplerConfig):
        super().__init__(config)

    def autoregressive(
        self,
        decoder: "TransformerDecoder | DecentralizedMlpDecoder",
        encoded_obs: Float[Tensor, "b agents emb"] | None = None,
        raw_obs: Float[Tensor, "b agents obs"] | None = None,
        available_actions: Tensor | None = None,
        deterministic: bool = False,
    ) -> AutoregressiveDiscreteSample:
        batch_size = get_batch_size(encoded_obs, raw_obs)
        actions = []
        log_probs = []
        if isinstance(decoder, TransformerDecoder):
            action_history = torch.zeros((batch_size, self.cfg.num_agents, self.cfg.act_dim + 1)).to(**self.tprops)
            action_history[:, 0, 0] = self.cfg.start_token
            for i in range(self.cfg.num_agents):
                logits = decoder(action_history, encoded_obs)[:, i, :]  # (b, action_dim)
                action, log_prob = self._sample_discrete_action(logits, available_actions, i, deterministic)
                actions.append(action)
                log_probs.append(log_prob)

                if i + 1 < self.cfg.num_agents:
                    action_history[:, i + 1, 1:] = F.one_hot(action, num_classes=self.cfg.act_dim)
        else:
            out = decoder(raw_obs)  # independent sampling for each agent
            for i in range(self.cfg.num_agents):
                logits = out[:, i, :]
                action, log_prob = self._sample_discrete_action(logits, available_actions, i, deterministic)
                actions.append(action)
                log_probs.append(log_prob)
        return AutoregressiveDiscreteSample(actions=torch.stack(actions, dim=1), log_probs=torch.stack(log_probs, dim=1))

    def parallel(
        self,
        actions: Float[Tensor, "b agents act"],
        decoder: "TransformerDecoder | DecentralizedMlpDecoder",
        encoded_obs: Float[Tensor, "b agents emb"] | None = None,
        raw_obs: Float[Tensor, "b agents obs"] | None = None,
        available_actions: Tensor | None = None,
    ) -> ParallelDiscreteSample:
        batch_size = get_batch_size(encoded_obs, raw_obs)

        if isinstance(decoder, DecentralizedMlpDecoder):
            logits = decoder(raw_obs)
        else:
            one_hot_action = F.one_hot(actions.long().squeeze(), num_classes=self.cfg.act_dim)  # (b, num_agents, action_dim)
            shifted_action = torch.zeros((batch_size, self.cfg.num_agents, self.cfg.act_dim + 1)).to(**self.tprops)
            shifted_action[:, 0, 0] = self.cfg.start_token
            shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]  # => [1, 0, ... 0], [0, agent_i_onehot]
            logits = decoder(shifted_action, encoded_obs)

        if available_actions is not None:
            logits[available_actions == 0] = -1e10

        dist = Categorical(logits=logits)
        return ParallelDiscreteSample(actions=dist.log_prob(actions.squeeze(-1)), entropy=dist.entropy())

    @staticmethod
    def _sample_discrete_action(
        logits: Float[Tensor, "b act_dim"],
        available_actions: Float[Tensor, "b agents actions"] | None,
        agent_idx: int,
        deterministic: bool,
    ) -> tuple[Float[Tensor, "b"], Float[Tensor, "b"]]:
        if available_actions is not None:
            logits[available_actions[:, agent_idx, :] == 0] = -1e10

        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action)


class ContinuousAutoregressiveSample(NamedTuple):
    actions: Float[Tensor, "b agents act"]
    log_probs: Float[Tensor, "b agents act"]


class ContinuousParallelSample(NamedTuple):
    actions: Float[Tensor, "b agents act"]
    entropy: Float[Tensor, "b"]


@dataclass
class ContinousSamplerConfig(SamplerConfig):
    std_scale: float = 0.5


class ContinuousSampler(Sampler):
    cfg: ContinousSamplerConfig

    def __init__(self, config: ContinousSamplerConfig):
        super().__init__(config)

    def autoregressive(
        self,
        decoder: TransformerDecoder | DecentralizedMlpDecoder,
        encoded_obs: Float[Tensor, "b agents emb"] | None = None,
        raw_obs: Float[Tensor, "b agents obs"] | None = None,
        deterministic: bool = False,
    ) -> ContinuousAutoregressiveSample:
        batch_size = get_batch_size(encoded_obs, raw_obs)
        actions = []
        log_probs = []
        action_history = torch.zeros((batch_size, self.cfg.num_agents, self.cfg.act_dim)).to(**self.tprops)
        for i in range(self.cfg.num_agents):
            act_mean = decoder(action_history, encoded_obs) if isinstance(decoder, TransformerDecoder) else decoder(raw_obs)
            act_mean = act_mean[:, i, :]

            action, dist = self._sample_continuous_action(act_mean, decoder.log_std, deterministic)
            actions.append(action)
            log_probs.append(dist.log_prob(action))

            if i + 1 < self.cfg.num_agents and isinstance(decoder, TransformerDecoder):
                action_history[:, i + 1, :] = action

        return ContinuousAutoregressiveSample(actions=torch.stack(actions, dim=1), log_probs=torch.stack(log_probs, dim=1))

    def parallel(
        self,
        actions: Float[Tensor, "b agents act"],
        decoder: "TransformerDecoder | DecentralizedMlpDecoder",
        encoded_obs: Float[Tensor, "b agents emb"] | None = None,
        raw_obs: Float[Tensor, "b agents obs"] | None = None,
        deterministic: bool = False,
    ) -> ContinuousParallelSample:
        batch_size = get_batch_size(encoded_obs, raw_obs)

        if isinstance(decoder, DecentralizedMlpDecoder):
            act_mean = decoder(raw_obs)
        else:
            shifted_action = torch.zeros((batch_size, self.cfg.num_agents, self.cfg.act_dim)).to(**self.tprops)
            shifted_action[:, 1:, :] = actions[:, :-1, :]
            act_mean = decoder(shifted_action, encoded_obs)

        actions, dist = self._sample_continuous_action(act_mean, decoder.log_std, deterministic)
        return ContinuousParallelSample(actions=actions, entropy=dist.entropy())

    def _sample_continuous_action(
        self, mean: Float[Tensor, "b act"], log_std: Float[Tensor, "act"], deterministic: bool
    ) -> tuple[Float[Tensor, "b act"], Normal]:
        std = torch.sigmoid(log_std) * self.cfg.std_scale
        dist = Normal(mean, std)
        action = mean if deterministic else dist.sample()
        return action, dist


def get_batch_size(encoded_obs: torch.Tensor | None, raw_obs: torch.Tensor | None) -> int:
    if encoded_obs is not None:
        return encoded_obs.size(0)
    if raw_obs is not None:
        return raw_obs.size(0)
    raise ValueError("Exactly one of encoded_obs or raw_obs must be provided.")
