import abc
from dataclasses import dataclass
from typing import NamedTuple, TypedDict

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.distributions import Categorical, Normal

from mat.decoder import DecentralizedMlpDecoder, TransformerDecoder


class TensorProps(TypedDict):
    device: torch.device
    dtype: torch.dtype


@dataclass
class SamplerConfig:
    batch_size: int
    num_agents: int
    act_dim: int
    tprops: TensorProps


class Sampler(abc.ABC):
    def __init__(self, config: SamplerConfig):
        self.cfg = config


class SamplerResult(NamedTuple):
    actions: Float[Tensor, "b agents 1"]
    log_probs: Float[Tensor, "b agents 1"]


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
    ) -> SamplerResult:
        assert bool(encoded_obs) != bool(raw_obs), "Exactly one of encoded_obs and raw_obs must be provided."

        actions = []
        log_probs = []
        if isinstance(decoder, TransformerDecoder):
            action_history = torch.zeros((self.cfg.batch_size, self.cfg.num_agents, self.cfg.act_dim + 1)).to(**self.cfg.tprops)
            action_history[:, 0, 0] = self.cfg.start_token
            for i in range(self.cfg.num_agents):
                logits = decoder(action_history, encoded_obs)[:, i, :]  # (b, action_dim)
                action, log_prob = self._sample_discrete_action(logits, available_actions, i, deterministic)
                actions.append(action)
                log_probs.append(log_prob)

                if i + 1 < self.cfg.num_agents:
                    action_history[:, i + 1, 1:] = F.one_hot(action.squeeze(-1), num_classes=self.cfg.act_dim)
        else:
            out = decoder(raw_obs)  # independent sampling for each agent
            for i in range(self.cfg.num_agents):
                logits = out[:, i, :]
                action, log_prob = self._sample_discrete_action(logits, available_actions, i, deterministic)
                actions.append(action)
                log_probs.append(log_prob)
        return SamplerResult(actions=torch.cat(actions, dim=1), log_probs=torch.cat(log_probs, dim=1))

    def parallel(
        self,
        actions: Float[Tensor, "b agents act"],
        decoder: "TransformerDecoder | DecentralizedMlpDecoder",
        encoded_obs: Float[Tensor, "b agents emb"] | None = None,
        raw_obs: Float[Tensor, "b agents obs"] | None = None,
        available_actions: Tensor | None = None,
    ) -> SamplerResult:
        assert bool(encoded_obs) != bool(raw_obs), "Exactly one of encoded_obs and raw_obs must be provided."

        if isinstance(decoder, DecentralizedMlpDecoder):
            logits = decoder(raw_obs)
        else:
            one_hot_action = F.one_hot(actions.squeeze(-1), num_classes=self.cfg.act_dim)  # (b, num_agents, action_dim)
            shifted_action = torch.zeros((self.cfg.batch_size, self.cfg.num_agents, self.cfg.act_dim + 1)).to(**self.cfg.tprops)
            shifted_action[:, 0, 0] = self.cfg.start_token
            shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]  # => [1, 0, ... 0], [0, agent_i_onehot]
            logits = decoder(shifted_action, encoded_obs)

        if available_actions is not None:
            logits[available_actions == 0] = -1e10

        dist = Categorical(logits=logits)
        return SamplerResult(actions=dist.log_prob(actions.squeeze(-1)).unsqueeze(-1), log_probs=dist.entropy().unsqueeze(-1))

    @staticmethod
    def _sample_discrete_action(
        logits: Float[Tensor, "b act_dim"],
        available_actions: Float[Tensor, "b agents actions"] | None,
        agent_idx: int,
        deterministic: bool,
    ) -> tuple[Float[Tensor, "b 1"], Float[Tensor, "b 1"]]:
        if available_actions is not None:
            logits[available_actions[:, agent_idx, :] == 0] = -1e10

        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return action.unsqueeze(-1), dist.log_prob(action).unsqueeze(-1)


class ContinuousSamplerResult(NamedTuple):
    actions: Float[Tensor, "b agents act"]
    log_probs: Float[Tensor, "b agents act"]


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
    ) -> ContinuousSamplerResult:
        assert bool(encoded_obs) != bool(raw_obs), "Exactly one of encoded_obs and raw_obs must be provided."

        actions = []
        log_probs = []
        action_history = torch.zeros((self.cfg.batch_size, self.cfg.num_agents, self.cfg.act_dim)).to(**self.cfg.tprops)
        for i in range(self.cfg.num_agents):
            act_mean = decoder(action_history, encoded_obs) if isinstance(decoder, TransformerDecoder) else decoder(raw_obs)
            act_mean = act_mean[:, i, :]

            action, log_prob = self._sample_continuous_action(act_mean, decoder.log_std, deterministic)
            actions.append(action)
            log_probs.append(log_prob)

            if i + 1 < self.cfg.num_agents and isinstance(decoder, TransformerDecoder):
                action_history[:, i + 1, :] = action

        return ContinuousSamplerResult(actions=torch.stack(actions, dim=1), log_probs=torch.stack(log_probs, dim=1))

    def parallel(
        self,
        actions: Float[Tensor, "b agents act"],
        decoder: "TransformerDecoder | DecentralizedMlpDecoder",
        encoded_obs: Float[Tensor, "b agents emb"] | None = None,
        raw_obs: Float[Tensor, "b agents obs"] | None = None,
        deterministic: bool = False,
    ) -> ContinuousSamplerResult:
        assert bool(encoded_obs) != bool(raw_obs), "Exactly one of encoded_obs and raw_obs must be provided."

        if isinstance(decoder, DecentralizedMlpDecoder):
            act_mean = decoder(raw_obs)
        else:
            shifted_action = torch.zeros((self.cfg.batch_size, self.cfg.num_agents, self.cfg.act_dim)).to(**self.cfg.tprops)
            shifted_action[:, 1:, :] = actions[:, :-1, :]
            act_mean = decoder(shifted_action, encoded_obs)

        actions, log_probs = self._sample_continuous_action(act_mean, decoder.log_std, deterministic)
        return ContinuousSamplerResult(actions=actions, log_probs=log_probs)

    def _sample_continuous_action(
        self, mean: Float[Tensor, "b act_dim"], log_std: Float[Tensor, "act_dim"], deterministic: bool
    ) -> tuple[Float[Tensor, "b act_dim"], Float[Tensor, "b act_dim"]]:
        std = torch.sigmoid(log_std) * self.cfg.std_scale
        dist = Normal(mean, std)
        action = mean if deterministic else dist.sample()
        return action, dist.log_prob(action)
