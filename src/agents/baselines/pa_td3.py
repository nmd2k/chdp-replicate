"""Parameterized Action TD3 (from PADDPG) for hybrid action spaces."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any


class PANetwork(nn.Module):
    """Actor-Critic network for PA-TD3."""

    def __init__(
        self,
        state_dim: int,
        discrete_action_dim: int,
        continuous_param_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.discrete_action_dim = discrete_action_dim
        self.continuous_param_dim = continuous_param_dim

        # Actor: state -> (discrete logits, continuous params)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.discrete_head = nn.Linear(hidden_dim, discrete_action_dim)
        self.continuous_head = nn.Linear(hidden_dim, continuous_param_dim)

        # Critic: (state, discrete_onehot, continuous_params) -> Q
        critic_input = state_dim + discrete_action_dim + continuous_param_dim
        self.critic = nn.Sequential(
            nn.Linear(critic_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_actor(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get discrete logits and continuous parameters."""
        features = self.actor(state)
        discrete_logits = self.discrete_head(features)
        continuous_params = self.continuous_head(features)
        return discrete_logits, continuous_params

    def forward_critic(
        self, 
        state: torch.Tensor, 
        discrete_action: torch.Tensor,
        continuous_params: torch.Tensor
    ) -> torch.Tensor:
        """Compute Q-value for given state and hybrid action."""
        discrete_onehot = F.one_hot(discrete_action.long(), self.discrete_action_dim).float()
        x = torch.cat([state, discrete_onehot, continuous_params], dim=-1)
        return self.critic(x).squeeze(-1)


class PATD3:
    """Parameterized Action TD3 algorithm."""

    def __init__(
        self,
        state_dim: int,
        discrete_action_dim: int,
        continuous_param_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
    ):
        self.discrete_action_dim = discrete_action_dim
        self.continuous_param_dim = continuous_param_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        # Actor-Critic networks
        self.actor = PANetwork(state_dim, discrete_action_dim, continuous_param_dim, hidden_dim)
        self.critic = PANetwork(state_dim, discrete_action_dim, continuous_param_dim, hidden_dim)
        self.critic_target = PANetwork(state_dim, discrete_action_dim, continuous_param_dim, hidden_dim)

        # Copy parameters to target
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.total_steps = 0

    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[int, np.ndarray]:
        """Select hybrid action."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            discrete_logits, continuous_params = self.actor.forward_actor(state_tensor)

            if explore:
                # Add noise to continuous params
                continuous_params += torch.randn_like(continuous_params) * 0.1

            # Select discrete action (argmax or sample)
            discrete_action = torch.argmax(discrete_logits, dim=-1).item()
            continuous_params = continuous_params[0].numpy()

        return discrete_action, continuous_params

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train PA-TD3 on a batch."""
        self.total_steps += 1

        states = batch["state"]
        discrete_actions = batch["discrete_action"]
        continuous_params = batch["continuous_params"]
        rewards = batch["reward"]
        next_states = batch["next_state"]
        dones = batch["done"]

        # Compute target Q
        with torch.no_grad():
            next_logits, next_cont = self.critic_target.forward_actor(next_states)
            if self.policy_noise > 0:
                noise = torch.randn_like(next_cont) * self.policy_noise
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                next_cont += noise

            next_discrete = torch.argmax(next_logits, dim=-1)
            next_q = self.critic_target.forward_critic(next_states, next_discrete, next_cont)
            target_q = rewards + self.gamma * (1 - dones) * next_q

        # Critic loss
        current_q = self.critic.forward_critic(states, discrete_actions, continuous_params)
        critic_loss = F.mse_loss(current_q, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update (delayed)
        actor_loss = None
        if self.total_steps % self.policy_freq == 0:
            # Actor: maximize Q
            logits, cont_params = self.actor.forward_actor(states)
            discrete = torch.argmax(logits, dim=-1)
            actor_loss = -self.critic.forward_critic(states, discrete, cont_params).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else 0.0,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict."""
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
