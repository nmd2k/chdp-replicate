"""Hierarchical Hybrid Q-Network with TD3 for hybrid action spaces."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any


class DiscreteQNetwork(nn.Module):
    """Q-network for discrete action selection."""

    def __init__(self, state_dim: int, discrete_action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, discrete_action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for discrete actions."""
        return self.net(state)


class ContinuousQNetwork(nn.Module):
    """Q-network for continuous parameter selection (conditioned on discrete)."""

    def __init__(
        self,
        state_dim: int,
        discrete_action_dim: int,
        continuous_param_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        # Input: state + discrete_onehot + continuous_params
        input_dim = state_dim + discrete_action_dim + continuous_param_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
        """Compute Q-value for given hybrid action."""
        discrete_onehot = F.one_hot(discrete_action.long(), self.discrete_action_dim).float()
        x = torch.cat([state, discrete_onehot, continuous_params], dim=-1)
        return self.net(x).squeeze(-1)


class ContinuousActor(nn.Module):
    """Actor for continuous parameters (conditioned on discrete action)."""

    def __init__(
        self,
        state_dim: int,
        discrete_action_dim: int,
        continuous_param_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.discrete_action_dim = discrete_action_dim
        # Input: state + discrete_onehot
        input_dim = state_dim + discrete_action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, continuous_param_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        discrete_action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute continuous parameters for given state and discrete action."""
        discrete_onehot = F.one_hot(discrete_action.long(), self.discrete_action_dim).float()
        x = torch.cat([state, discrete_onehot], dim=-1)
        return self.net(x)


class HHQNTD3:
    """Hierarchical Hybrid Q-Network with TD3."""

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

        # Discrete level
        self.discrete_q = DiscreteQNetwork(state_dim, discrete_action_dim, hidden_dim)
        self.discrete_q_target = DiscreteQNetwork(state_dim, discrete_action_dim, hidden_dim)
        self.discrete_q_target.load_state_dict(self.discrete_q.state_dict())

        # Continuous level
        self.continuous_q = ContinuousQNetwork(
            state_dim, discrete_action_dim, continuous_param_dim, hidden_dim
        )
        self.continuous_q_target = ContinuousQNetwork(
            state_dim, discrete_action_dim, continuous_param_dim, hidden_dim
        )
        self.continuous_q_target.load_state_dict(self.continuous_q.state_dict())

        self.continuous_actor = ContinuousActor(
            state_dim, discrete_action_dim, continuous_param_dim, hidden_dim
        )

        # Optimizers
        self.discrete_optimizer = torch.optim.Adam(self.discrete_q.parameters(), lr=lr)
        self.continuous_q_optimizer = torch.optim.Adam(self.continuous_q.parameters(), lr=lr)
        self.continuous_actor_optimizer = torch.optim.Adam(
            self.continuous_actor.parameters(), lr=lr
        )

        self.total_steps = 0

    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[int, np.ndarray]:
        """Select hybrid action hierarchically."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Level 1: Select discrete action
            discrete_q_values = self.discrete_q(state_tensor)
            discrete_action = torch.argmax(discrete_q_values, dim=-1).item()

            # Level 2: Select continuous parameters for chosen discrete action
            d_tensor = torch.tensor([discrete_action]).long()
            continuous_params = self.continuous_actor.forward(state_tensor, d_tensor)

            if explore:
                continuous_params += torch.randn_like(continuous_params) * 0.1

        return discrete_action, continuous_params[0].numpy()

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train HHQN-TD3 on a batch."""
        self.total_steps += 1

        states = batch["state"]
        discrete_actions = batch["discrete_action"]
        continuous_params = batch["continuous_params"]
        rewards = batch["reward"]
        next_states = batch["next_state"]
        dones = batch["done"]

        # Continuous level update
        with torch.no_grad():
            # Get next discrete actions
            next_discrete_q = self.discrete_q_target(next_states)
            next_discrete = torch.argmax(next_discrete_q, dim=-1)

            # Get next continuous params
            next_cont = self.continuous_actor.forward(next_states, next_discrete)
            if self.policy_noise > 0:
                noise = torch.randn_like(next_cont) * self.policy_noise
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                next_cont += noise

            # Target Q for continuous level
            next_q = self.continuous_q_target.forward(next_states, next_discrete, next_cont)
            target_q = rewards + self.gamma * (1 - dones) * next_q

        # Continuous Q loss
        current_q = self.continuous_q.forward(states, discrete_actions, continuous_params)
        cont_q_loss = F.mse_loss(current_q, target_q)

        self.continuous_q_optimizer.zero_grad()
        cont_q_loss.backward()
        self.continuous_q_optimizer.step()

        # Continuous actor update
        cont_actor_loss = None
        if self.total_steps % self.policy_freq == 0:
            pred_cont = self.continuous_actor.forward(states, discrete_actions)
            cont_actor_loss = -self.continuous_q.forward(
                states, discrete_actions, pred_cont
            ).mean()

            self.continuous_actor_optimizer.zero_grad()
            cont_actor_loss.backward()
            self.continuous_actor_optimizer.step()

        # Discrete level update
        discrete_q_pred = self.discrete_q(states)
        # Use continuous Q-values as targets for discrete level
        with torch.no_grad():
            cont_q_values = []
            for d in range(self.discrete_action_dim):
                d_tensor = torch.full_like(discrete_actions, d)
                cont_params = self.continuous_actor.forward(states, d_tensor)
                q = self.continuous_q.forward(states, d_tensor, cont_params)
                cont_q_values.append(q)
            discrete_targets = torch.stack(cont_q_values, dim=-1)

        discrete_loss = F.mse_loss(discrete_q_pred, discrete_targets)

        self.discrete_optimizer.zero_grad()
        discrete_loss.backward()
        self.discrete_optimizer.step()

        # Soft update targets
        if self.total_steps % self.policy_freq == 0:
            for param, target_param in zip(
                self.discrete_q.parameters(), self.discrete_q_target.parameters()
            ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(
                self.continuous_q.parameters(), self.continuous_q_target.parameters()
            ):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "discrete_loss": discrete_loss.item(),
            "cont_q_loss": cont_q_loss.item(),
            "cont_actor_loss": cont_actor_loss.item() if cont_actor_loss else 0.0,
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict."""
        return {
            "discrete_q": self.discrete_q.state_dict(),
            "discrete_q_target": self.discrete_q_target.state_dict(),
            "continuous_q": self.continuous_q.state_dict(),
            "continuous_q_target": self.continuous_q_target.state_dict(),
            "continuous_actor": self.continuous_actor.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict."""
        self.discrete_q.load_state_dict(state_dict["discrete_q"])
        self.discrete_q_target.load_state_dict(state_dict["discrete_q_target"])
        self.continuous_q.load_state_dict(state_dict["continuous_q"])
        self.continuous_q_target.load_state_dict(state_dict["continuous_q_target"])
        self.continuous_actor.load_state_dict(state_dict["continuous_actor"])
