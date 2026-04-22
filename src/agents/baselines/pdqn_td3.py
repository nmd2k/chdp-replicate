"""Parameterized DQN with TD3 for hybrid action spaces."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any


class PDQNNetwork(nn.Module):
    """Q-network for PDQN that takes hybrid actions."""

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

        # Q-network: (state, discrete_action_onehot, continuous_params) -> Q
        input_dim = state_dim + discrete_action_dim + continuous_param_dim
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, discrete_action: torch.Tensor, 
                continuous_params: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for given state and hybrid action."""
        # One-hot encode discrete action
        discrete_onehot = F.one_hot(discrete_action.long(), self.discrete_action_dim).float()
        
        # Concatenate state, discrete action, and continuous params
        x = torch.cat([state, discrete_onehot, continuous_params], dim=-1)
        return self.q_net(x).squeeze(-1)


class PDQNTD3:
    """PDQN with TD3 algorithm for hybrid action spaces."""

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

        # Actor: state -> continuous parameters (for each discrete action)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, continuous_param_dim * discrete_action_dim),
        )

        # Two Q-networks for TD3
        self.q1 = PDQNNetwork(state_dim, discrete_action_dim, continuous_param_dim, hidden_dim)
        self.q2 = PDQNNetwork(state_dim, discrete_action_dim, continuous_param_dim, hidden_dim)
        self.q1_target = PDQNNetwork(state_dim, discrete_action_dim, continuous_param_dim, hidden_dim)
        self.q2_target = PDQNNetwork(state_dim, discrete_action_dim, continuous_param_dim, hidden_dim)

        # Copy parameters to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)

        self.total_steps = 0

    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[int, np.ndarray]:
        """Select hybrid action (discrete + continuous)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get continuous parameters for all discrete actions
            cont_params = self.actor(state_tensor).view(
                self.discrete_action_dim, self.continuous_param_dim
            )

            if explore:
                cont_params += torch.randn_like(cont_params) * 0.1

            # Evaluate Q for each discrete action with its continuous params
            q_values = []
            for d in range(self.discrete_action_dim):
                d_tensor = torch.tensor([d]).long()
                q = self.q1(state_tensor.repeat(self.discrete_action_dim, 1), 
                           torch.arange(self.discrete_action_dim), cont_params)
                q_values.append(q[d])

            # Select best discrete action
            discrete_action = torch.argmax(torch.stack(q_values)).item()
            continuous_params = cont_params[discrete_action].numpy()

        return discrete_action, continuous_params

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train PDQN-TD3 on a batch of experiences."""
        self.total_steps += 1

        states = batch["state"]
        discrete_actions = batch["discrete_action"]
        continuous_params = batch["continuous_params"]
        rewards = batch["reward"]
        next_states = batch["next_state"]
        dones = batch["done"]

        # Select next actions
        with torch.no_grad():
            next_cont_params = self.actor(next_states)
            if self.policy_noise > 0:
                noise = torch.randn_like(next_cont_params) * self.policy_noise
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                next_cont_params += noise

            # Compute target Q-values
            next_q1 = self.q1_target(next_states, discrete_actions, next_cont_params)
            next_q2 = self.q2_target(next_states, discrete_actions, next_cont_params)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + self.gamma * (1 - dones) * next_q

        # Critic loss
        q1_pred = self.q1(states, discrete_actions, continuous_params)
        q2_pred = self.q2(states, discrete_actions, continuous_params)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        q_loss = q1_loss + q2_loss

        # Update critics
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Actor update (delayed)
        actor_loss = None
        if self.total_steps % self.policy_freq == 0:
            # Update actor to maximize Q1
            cont_params = self.actor(states)
            actor_loss = -self.q1(states, discrete_actions, cont_params).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else 0.0,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving."""
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict."""
        self.actor.load_state_dict(state_dict["actor"])
        self.q1.load_state_dict(state_dict["q1"])
        self.q2.load_state_dict(state_dict["q2"])
        self.q1_target.load_state_dict(state_dict["q1_target"])
        self.q2_target.load_state_dict(state_dict["q2_target"])
