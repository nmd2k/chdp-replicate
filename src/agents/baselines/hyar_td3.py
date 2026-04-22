"""Hybrid Action Representation with TD3 (HyAR-TD3) - prior SOTA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any


class HyAREncoder(nn.Module):
    """Encoder for hybrid action representation."""

    def __init__(
        self,
        state_dim: int,
        discrete_action_dim: int,
        continuous_param_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.discrete_action_dim = discrete_action_dim
        self.continuous_param_dim = continuous_param_dim
        self.latent_dim = latent_dim

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Action encoder (discrete + continuous)
        action_input_dim = discrete_action_dim + continuous_param_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(action_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        discrete_action: torch.Tensor,
        continuous_params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode state and action into latent space."""
        state_latent = self.state_encoder(state)

        # One-hot encode discrete action
        discrete_onehot = F.one_hot(discrete_action.long(), self.discrete_action_dim).float()
        action_input = torch.cat([discrete_onehot, continuous_params], dim=-1)
        action_latent = self.action_encoder(action_input)

        return state_latent, action_latent


class HyARActor(nn.Module):
    """Deterministic actor for HyAR."""

    def __init__(
        self,
        state_dim: int,
        discrete_action_dim: int,
        continuous_param_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.discrete_action_dim = discrete_action_dim
        self.continuous_param_dim = continuous_param_dim

        # State to latent
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Latent to discrete logits
        self.discrete_head = nn.Linear(hidden_dim, discrete_action_dim)

        # Latent + discrete to continuous
        cont_input_dim = hidden_dim + discrete_action_dim
        self.continuous_head = nn.Sequential(
            nn.Linear(cont_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, continuous_param_dim),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get deterministic hybrid action."""
        features = self.state_net(state)

        # Discrete action (deterministic via argmax during inference)
        discrete_logits = self.discrete_head(features)

        # Continuous parameters
        # During training, use sampled discrete; during inference, use argmax
        discrete = torch.argmax(discrete_logits, dim=-1)
        discrete_onehot = F.one_hot(discrete.long(), self.discrete_action_dim).float()

        cont_input = torch.cat([features, discrete_onehot], dim=-1)
        continuous_params = self.continuous_head(cont_input)

        return discrete_logits, continuous_params


class HyARCritic(nn.Module):
    """Critic for HyAR operating in latent space."""

    def __init__(
        self,
        state_dim: int,
        discrete_action_dim: int,
        continuous_param_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = HyAREncoder(
            state_dim, discrete_action_dim, continuous_param_dim, latent_dim, hidden_dim
        )

        # Q-network on concatenated latents
        self.q_net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        discrete_action: torch.Tensor,
        continuous_params: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q-value."""
        state_latent, action_latent = self.encoder(state, discrete_action, continuous_params)
        x = torch.cat([state_latent, action_latent], dim=-1)
        return self.q_net(x).squeeze(-1)


class HyARTD3:
    """HyAR with TD3 algorithm (prior SOTA)."""

    def __init__(
        self,
        state_dim: int,
        discrete_action_dim: int,
        continuous_param_dim: int,
        latent_dim: int = 64,
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

        # Actor and Critic
        self.actor = HyARActor(
            state_dim, discrete_action_dim, continuous_param_dim, latent_dim, hidden_dim
        )
        self.critic = HyARCritic(
            state_dim, discrete_action_dim, continuous_param_dim, latent_dim, hidden_dim
        )
        self.critic_target = HyARCritic(
            state_dim, discrete_action_dim, continuous_param_dim, latent_dim, hidden_dim
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.total_steps = 0

    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[int, np.ndarray]:
        """Select hybrid action."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            discrete_logits, continuous_params = self.actor.forward(state_tensor)

            if explore:
                continuous_params += torch.randn_like(continuous_params) * 0.1

            discrete_action = torch.argmax(discrete_logits, dim=-1).item()
            continuous_params = continuous_params[0].numpy()

        return discrete_action, continuous_params

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train HyAR-TD3 on a batch."""
        self.total_steps += 1

        states = batch["state"]
        discrete_actions = batch["discrete_action"]
        continuous_params = batch["continuous_params"]
        rewards = batch["reward"]
        next_states = batch["next_state"]
        dones = batch["done"]

        # Compute target Q
        with torch.no_grad():
            next_logits, next_cont = self.actor.forward(next_states)
            if self.policy_noise > 0:
                noise = torch.randn_like(next_cont) * self.policy_noise
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                next_cont += noise

            next_discrete = torch.argmax(next_logits, dim=-1)
            next_q = self.critic_target.forward(next_states, next_discrete, next_cont)
            target_q = rewards + self.gamma * (1 - dones) * next_q

        # Critic loss
        current_q = self.critic.forward(states, discrete_actions, continuous_params)
        critic_loss = F.mse_loss(current_q, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update (delayed)
        actor_loss = None
        if self.total_steps % self.policy_freq == 0:
            # Actor: maximize Q
            logits, cont_params = self.actor.forward(states)
            discrete = torch.argmax(logits, dim=-1)
            actor_loss = -self.critic.forward(states, discrete, cont_params).mean()

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
