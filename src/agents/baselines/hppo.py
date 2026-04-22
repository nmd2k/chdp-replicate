"""Hybrid PPO for hybrid action spaces."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any


class HPPOPolicy(nn.Module):
    """Actor-Critic policy for HPPO."""

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

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Discrete head (categorical)
        self.discrete_head = nn.Linear(hidden_dim, discrete_action_dim)

        # Continuous head (Gaussian mean)
        self.continuous_mean = nn.Linear(hidden_dim, continuous_param_dim)
        self.continuous_logstd = nn.Parameter(torch.zeros(continuous_param_dim))

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning policy outputs and value."""
        features = self.backbone(state)

        # Discrete action logits
        discrete_logits = self.discrete_head(features)

        # Continuous action parameters
        cont_mean = self.continuous_mean(features)
        cont_logstd = self.continuous_logstd.expand_as(cont_mean)
        cont_std = torch.exp(cont_logstd)

        # Value
        value = self.value_head(features)

        return discrete_logits, cont_mean, cont_std, value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        discrete_logits, cont_mean, cont_std, _ = self.forward(state)

        # Discrete action
        if deterministic:
            discrete_action = torch.argmax(discrete_logits, dim=-1)
        else:
            discrete_dist = torch.distributions.Categorical(logits=discrete_logits)
            discrete_action = discrete_dist.sample()

        # Continuous action
        if deterministic:
            continuous_action = cont_mean
        else:
            cont_dist = torch.distributions.Normal(cont_mean, cont_std)
            continuous_action = cont_dist.sample()

        # Log probability
        discrete_logp = F.log_softmax(discrete_logits, dim=-1)
        discrete_logp = discrete_logp.gather(1, discrete_action.unsqueeze(-1)).squeeze(-1)

        cont_logp = self._gaussian_logprob(cont_mean, cont_std, continuous_action)
        cont_logp = cont_logp.sum(dim=-1)

        logp = discrete_logp + cont_logp

        return discrete_action, continuous_action, logp

    def evaluate(
        self,
        state: torch.Tensor,
        discrete_action: torch.Tensor,
        continuous_action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy of given actions."""
        discrete_logits, cont_mean, cont_std, value = self.forward(state)

        # Discrete log prob
        discrete_logp = F.log_softmax(discrete_logits, dim=-1)
        discrete_logp = discrete_logp.gather(1, discrete_action.unsqueeze(-1)).squeeze(-1)

        # Continuous log prob
        cont_logp = self._gaussian_logprob(cont_mean, cont_std, continuous_action)
        cont_logp = cont_logp.sum(dim=-1)

        # Total log prob
        logp = discrete_logp + cont_logp

        # Entropy
        discrete_dist = torch.distributions.Categorical(logits=discrete_logits)
        discrete_entropy = discrete_dist.entropy().mean()

        cont_dist = torch.distributions.Normal(cont_mean, cont_std)
        cont_entropy = cont_dist.entropy().sum(dim=-1).mean()

        entropy = discrete_entropy + cont_entropy

        return logp, entropy

    def _gaussian_logprob(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of action under Gaussian."""
        var = std ** 2
        logp = -0.5 * (((action - mean) ** 2) / var + torch.log(2 * np.pi * var))
        return logp


class HPPO:
    """Hybrid PPO algorithm."""

    def __init__(
        self,
        state_dim: int,
        discrete_action_dim: int,
        continuous_param_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        lr: float = 3e-4,
        epochs: int = 10,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        self.discrete_action_dim = discrete_action_dim
        self.continuous_param_dim = continuous_param_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Policy
        self.policy = HPPOPolicy(state_dim, discrete_action_dim, continuous_param_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, np.ndarray]:
        """Select hybrid action."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            discrete, continuous, _ = self.policy.get_action(state_tensor, deterministic)

        return discrete.item(), continuous[0].numpy()

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.stack(advantages)
        returns = advantages + values
        return advantages, returns

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train HPPO on a batch of trajectories."""
        states = batch["state"]
        discrete_actions = batch["discrete_action"]
        continuous_actions = batch["continuous_params"]
        old_logps = batch["logp"]
        returns = batch["return"]
        advantages = batch["advantage"]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.epochs):
            # Evaluate current policy
            logp, entropy = self.policy.evaluate(states, discrete_actions, continuous_actions)
            _, _, _, values = self.policy.forward(states)

            # Ratio
            ratio = torch.exp(logp - old_logps)

            # Surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)

            # Entropy bonus
            entropy_loss = -entropy

            # Total loss
            loss = (
                policy_loss
                + self.value_coef * value_loss
                + self.entropy_coef * entropy_loss
            )

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": (old_logps - logp).mean().item(),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict."""
        return {"policy": self.policy.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict."""
        self.policy.load_state_dict(state_dict["policy"])
