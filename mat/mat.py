from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor, nn

from mat.decoder import DecentralizedMlpDecoder, TransformerDecoder
from mat.encoder import Encoder
from mat.samplers import ContinuousSampler, DiscreteSampler


@dataclass(frozen=True, slots=True, kw_only=True)
class MATInferenceOutput:
    actions: Float[Tensor, "b agents act"]
    action_log_probs: Float[Tensor, "b agents act"]
    values: Float[Tensor, "b agents 1"]


@dataclass(frozen=True, slots=True, kw_only=True)  # NamedTuple doesn't support inheritance
class MATTrainingOutput(MATInferenceOutput):
    entropy: Float[Tensor, "b"]


class MAT(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: TransformerDecoder | DecentralizedMlpDecoder,
        sampler: DiscreteSampler | ContinuousSampler,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.use_encoded_obs = isinstance(decoder, TransformerDecoder)
        self.use_raw_obs = isinstance(decoder, DecentralizedMlpDecoder)
        self.is_discrete = isinstance(sampler, DiscreteSampler)

    def forward(
        self,
        obs: Float[Tensor, "b agents obs"],
        actions: Float[Tensor, "b agents act"],
        available_actions: Float[Tensor, "b agents act"] | None = None,
    ) -> MATTrainingOutput:
        values, encoded_obs = self.encoder(obs)
        kwargs = self._get_sampler_kwargs(encoded_obs=encoded_obs, obs=obs, available_actions=available_actions)
        action_log_probs, entropy = self.sampler.parallel(actions=actions, decoder=self.decoder, **kwargs)
        return MATTrainingOutput(actions=actions, action_log_probs=action_log_probs, values=values, entropy=entropy)

    def get_actions(
        self,
        obs: Float[Tensor, "b agents obs"],
        available_actions: Float[Tensor, "b agents act"] | None = None,
        deterministic: bool = False,
    ) -> MATInferenceOutput:
        values, encoded_obs = self.encoder(obs)
        kwargs = self._get_sampler_kwargs(encoded_obs=encoded_obs, obs=obs, available_actions=available_actions)
        actions, action_log_probs = self.sampler.autoregressive(decoder=self.decoder, deterministic=deterministic, **kwargs)
        return MATInferenceOutput(actions=actions, action_log_probs=action_log_probs, values=values)

    def get_values(self, obs: Float[Tensor, "b agents obs"]) -> Float[Tensor, "b agents 1"]:
        values, _ = self.encoder(obs)
        return values

    def _get_sampler_kwargs(self, encoded_obs: Tensor, obs: Tensor, available_actions: Tensor | None = None) -> dict:
        return {
            "encoded_obs": encoded_obs if self.use_encoded_obs else None,
            "raw_obs": obs if self.use_raw_obs else None,
        } | ({"available_actions": available_actions} if self.is_discrete else {})
