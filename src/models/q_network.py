"""
Q-Network Critics for CHDP

Implements:
- Double Q-learning architecture (Eq. 3-4)
- Twin Q-networks for reducing overestimation bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import copy


class QNetwork(nn.Module):
    """
    Single Q-network critic.
    
    Estimates Q(s, a) for state-action pairs.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        input_dim = state_dim + action_dim
        
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value for state-action pair.
        
        Args:
            state: State [batch, state_dim] or [batch, 1, state_dim]
            action: Action [batch, action_dim] or [batch, 1, action_dim]
            
        Returns:
            q_value: Q-value [batch, 1] or [batch, K, 1]
        """
        if state.dim() == 3:
            batch_size, num_actions, _ = state.shape
            state = state.view(-1, self.state_dim)
            action = action.view(-1, self.action_dim)
            
            x = torch.cat([state, action], dim=-1)
            x = self.backbone(x)
            q_value = self.output(x)
            
            return q_value.view(batch_size, num_actions, 1)
        else:
            x = torch.cat([state, action], dim=-1)
            x = self.backbone(x)
            q_value = self.output(x)
            return q_value


class DoubleQNetwork(nn.Module):
    """
    Double Q-learning architecture with twin critics.
    
    Maintains two Q-networks to reduce overestimation bias.
    Uses the minimum of the two Q-values for target computation.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim, num_layers)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim, num_layers)
        
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)
        
        self._freeze_target_networks()
    
    def _freeze_target_networks(self):
        """Freeze target network parameters."""
        for param in self.target_q1.parameters():
            param.requires_grad = False
        for param in self.target_q2.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        use_target: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values from both critics.
        
        Args:
            state: State [batch, state_dim]
            action: Action [batch, action_dim]
            use_target: Use target networks instead of online
            
        Returns:
            q1: Q-value from first critic
            q2: Q-value from second critic
        """
        if use_target:
            q1 = self.target_q1(state, action)
            q2 = self.target_q2(state, action)
        else:
            q1 = self.q1(state, action)
            q2 = self.q2(state, action)
        
        return q1, q2
    
    def compute_min_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        use_target: bool = False
    ) -> torch.Tensor:
        """
        Compute minimum Q-value (for conservative target).
        
        Args:
            state: State [batch, state_dim]
            action: Action [batch, action_dim]
            use_target: Use target networks
            
        Returns:
            min_q: Minimum of q1 and q2
        """
        q1, q2 = self.forward(state, action, use_target)
        return torch.min(q1, q2)
    
    def update_target_networks(self, tau: float = 0.005):
        """
        Soft update target networks.
        
        θ_target ← τ × θ_online + (1-τ) × θ_target
        
        Args:
            tau: Target network update rate
        """
        for target_param, online_param in zip(
            self.target_q1.parameters(), self.q1.parameters()
        ):
            target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)
        
        for target_param, online_param in zip(
            self.target_q2.parameters(), self.q2.parameters()
        ):
            target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)
    
    def compute_q_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        gamma: float = 0.99
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Double Q-learning loss.
        
        L_q(θ) = E[(Q(s,a) - y)²] where y = r + γ(1-d) × min(Q_target(s', a'))
        
        Args:
            state: Current state [batch, state_dim]
            action: Current action [batch, action_dim]
            next_state: Next state [batch, state_dim]
            next_action: Next action [batch, action_dim]
            reward: Reward [batch, 1]
            done: Done flag [batch, 1]
            gamma: Discount factor
            
        Returns:
            loss: Combined Q-loss from both critics
            info: Loss information dict
        """
        with torch.no_grad():
            next_q1_target, next_q2_target = self.forward(
                next_state, next_action, use_target=True
            )
            next_q_min = torch.min(next_q1_target, next_q2_target)
            
            target = reward + gamma * (1 - done) * next_q_min
        
        q1, q2 = self.forward(state, action, use_target=False)
        
        loss1 = F.mse_loss(q1, target)
        loss2 = F.mse_loss(q2, target)
        loss = loss1 + loss2
        
        info = {
            'q1_loss': loss1.item(),
            'q2_loss': loss2.item(),
            'q1_mean': q1.mean().item(),
            'q2_mean': q2.mean().item(),
            'target_mean': target.mean().item()
        }
        
        return loss, info


class StateActionQNetwork(nn.Module):
    """
    Q-network that accepts state and latent action representation.
    
    Used for Q-guidance in codebook alignment.
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value for state-latent pair.
        
        Args:
            state: State [batch, state_dim] or [batch, K, state_dim]
            latent: Latent/codeword [batch, latent_dim] or [batch, K, latent_dim]
            
        Returns:
            q_value: Q-value [batch, 1] or [batch, K, 1]
        """
        if state.dim() == 3:
            batch_size, num_latents, _ = state.shape
            state = state.view(-1, self.state_dim)
            latent = latent.view(-1, self.latent_dim)
            
            x = torch.cat([state, latent], dim=-1)
            q_value = self.network(x)
            
            return q_value.view(batch_size, num_latents, 1)
        else:
            x = torch.cat([state, latent], dim=-1)
            q_value = self.network(x)
            return q_value
