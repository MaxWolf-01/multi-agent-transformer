import numpy as np
import supersuit as ss
import torch
from jaxtyping import Float
from pettingzoo.mpe import simple_spread_v3
from torch import Tensor

from mat.buffer import Buffer
from mat.mat import MAT
from mat.runners.base import EnvRunner


class MPERunner(EnvRunner):
    SUPPORTED_ENVS = {
        "simple_spread_v3": simple_spread_v3.parallel_env,
    }

    @staticmethod
    def get_env(env_id: str, env_kwargs: dict):
        if env_id not in MPERunner.SUPPORTED_ENVS:
            raise ValueError(f"Unknown MPE environment: {env_id}")
        return MPERunner.SUPPORTED_ENVS[env_id](**env_kwargs)

    def __init__(
        self,
        env_id: str,
        env_kwargs: dict,
        policy: MAT,
        buffer: Buffer,
        num_envs: int,
        device: str | torch.device,
        render: bool = False,
    ):
        """If render_kwargs is given, a separate env for rendering will be created."""
        super().__init__(device, buffer)
        env = self.get_env(env_id, env_kwargs)
        self.env = ss.concat_vec_envs_v1(ss.pettingzoo_env_to_vec_env_v1(env), num_vec_envs=num_envs)
        self.policy = policy
        self.buffer = buffer
        self.num_parallel_envs = num_envs
        self.device = device

        env.reset()  # required to access agents / num_agents attribute
        self.num_agents = env.num_agents  # not equal to self.env.agents, which is num_paralell_envs * num_agents

        self.next_obs = self._get_reshaped_obs_tensor(self.env.reset()[0])
        self.render_env = self.get_env(env_id=env_id, env_kwargs=env_kwargs | dict(render_mode="human")) if render else None
        self.render_obs, _ = self.render_env.reset() if render else (None, None)

    @torch.inference_mode()
    def collect_rollout(self) -> Float[Tensor, "b agents"]:
        """Collects a rollout using the current policy and returns value the estimate for final state."""
        for step in range(self.buffer.cfg.size):
            policy_output = self.policy.get_actions(obs=(obs := self.next_obs))
            actions = policy_output.actions.cpu().numpy()  # (batch, agents)
            self.next_obs, rewards, terminations, truncations, infos = self.env.step(actions.reshape(-1))
            self.next_obs = self._get_reshaped_obs_tensor(self.next_obs)
            self.buffer.insert(
                obs=obs.cpu().numpy(),
                actions=actions[:, :, None],
                action_log_probs=policy_output.action_log_probs.cpu().numpy(),
                values=policy_output.values.cpu().numpy(),
                rewards=rewards.reshape(self.num_parallel_envs, self.num_agents),
                dones=(terminations | truncations).reshape(self.num_parallel_envs, self.num_agents),
                active_masks=None,
            )
            if self.render_env is not None:
                self._render_step()
        return self.policy.get_values(self.next_obs)

    def _get_reshaped_obs_tensor(self, o: np.ndarray) -> Float[Tensor, "b agents *obs_shape"]:
        return torch.tensor(o.reshape(self.num_parallel_envs, self.num_agents, -1), device=self.device, dtype=torch.float32)

    def _render_step(self) -> None:
        obs = np.array([self.render_obs[agent] for agent in self.render_env.agents]).reshape(1, self.num_agents, -1)
        policy_output = self.policy.get_actions(obs=torch.tensor(obs, device=self.device))
        self.render_obs, _, terminations, truncations, _ = self.render_env.step(
            {agent: action for agent, action in zip(self.render_env.agents, policy_output.actions[0].cpu().numpy())}
        )
        if any(terminations.values()) or any(truncations.values()):  # pettingzoo doesn't automatically reset :/
            self.render_obs, _ = self.render_env.reset()
