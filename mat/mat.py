from typing import NamedTuple

from jaxtyping import Float
from torch import Tensor, nn

from mat.decoder import DecentralizedMlpDecoder, TransformerDecoder
from mat.encoder import Encoder
from mat.samplers import ContinuousSampler, DiscreteSampler


class MATOutput(NamedTuple):
    actions: Float[Tensor, "b agents act"] | Float[Tensor, "b agents 1"]
    action_log_probs: Float[Tensor, "b agents act"] | Float[Tensor, "b agents 1"]
    values: Float[Tensor, "b agents 1"]


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

    def forward(
        self,
        obs: Float[Tensor, "b agents obs"],
        actions: Float[Tensor, "b agents act"],
        available_actions: Float[Tensor, "b agents act"] | None = None,
    ) -> MATOutput:
        values, encoded_obs = self.encoder(obs)

        if isinstance(self.sampler, DiscreteSampler):
            action_log_probs, entropy = self.sampler.parallel(
                actions=actions,
                decoder=self.decoder,
                encoded_obs=encoded_obs if isinstance(self.decoder, TransformerDecoder) else None,
                raw_obs=obs if isinstance(self.decoder, DecentralizedMlpDecoder) else None,
                available_actions=available_actions,
            )
        else:
            action_log_probs, entropy = self.sampler.parallel(
                actions=actions,
                decoder=self.decoder,
                encoded_obs=encoded_obs if isinstance(self.decoder, TransformerDecoder) else None,
                raw_obs=obs if isinstance(self.decoder, DecentralizedMlpDecoder) else None,
            )

        return MATOutput(actions=actions, action_log_probs=action_log_probs, values=values)

    def get_actions(
        self,
        obs: Float[Tensor, "b agents obs"],
        available_actions: Float[Tensor, "b agents act"] | None = None,
        deterministic: bool = False,
    ) -> MATOutput:
        values, encoded_obs = self.encoder(obs)

        if isinstance(self.sampler, DiscreteSampler):
            actions, action_log_probs = self.sampler.autoregressive(
                available_actions=available_actions,
                decoder=self.decoder,
                encoded_obs=encoded_obs if isinstance(self.decoder, TransformerDecoder) else None,
                raw_obs=obs if isinstance(self.decoder, DecentralizedMlpDecoder) else None,
                deterministic=deterministic,
            )
        else:
            actions, action_log_probs = self.sampler.autoregressive(
                decoder=self.decoder,
                encoded_obs=encoded_obs if isinstance(self.decoder, TransformerDecoder) else None,
                raw_obs=obs if isinstance(self.decoder, DecentralizedMlpDecoder) else None,
                deterministic=deterministic,
            )
        return MATOutput(actions=actions, action_log_probs=action_log_probs, values=values)

    def get_values(self, obs: Float[Tensor, "b agents obs"]) -> Float[Tensor, "b agents 1"]:
        values, _ = self.encoder(obs)
        return values
