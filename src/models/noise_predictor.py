"""
Noise Prediction Networks for CHDP

Implements:
- ε_θd: Noise predictor for discrete policy
- ε_θc: Noise predictor for continuous policy (conditioned on codeword)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(-embeddings * torch.arange(half_dim, device=device))
        embeddings = time.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeConditioning(nn.Module):
    """Time conditioning module using FiLM."""
    
    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        return x * t_emb.unsqueeze(-1) + t_emb.unsqueeze(-1)


class DiscreteNoisePredictor(nn.Module):
    """
    ε_θd: Noise predictor for discrete policy.
    
    Predicts noise for discrete action space diffusion.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        num_layers: int = 3
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.time_embed = SinusoidalPositionEmbeddings(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        input_dim = action_dim + state_dim + hidden_dim
        
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(
        self,
        a_t: torch.Tensor,
        state: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict noise for discrete actions.
        
        Args:
            a_t: Noisy action [batch, action_dim]
            state: State [batch, state_dim]
            t: Timestep [batch]
            condition: Unused for discrete policy
            
        Returns:
            noise_pred: Predicted noise [batch, action_dim]
        """
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        
        x = torch.cat([a_t, state, t_emb], dim=-1)
        x = self.backbone(x)
        noise_pred = self.output(x)
        
        return noise_pred


class ContinuousNoisePredictor(nn.Module):
    """
    ε_θc: Noise predictor for continuous policy.
    
    Conditioned on codeword from Q-guided codebook.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        codebook_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        num_layers: int = 3
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.codebook_dim = codebook_dim
        self.hidden_dim = hidden_dim
        
        self.time_embed = SinusoidalPositionEmbeddings(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.codebook_proj = nn.Sequential(
            nn.Linear(codebook_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        input_dim = action_dim + state_dim + hidden_dim + hidden_dim
        
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(
        self,
        a_t: torch.Tensor,
        state: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise for continuous actions.
        
        Args:
            a_t: Noisy action [batch, action_dim]
            state: State [batch, state_dim]
            t: Timestep [batch]
            condition: Codeword from codebook [batch, codebook_dim]
            
        Returns:
            noise_pred: Predicted noise [batch, action_dim]
        """
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        
        c_emb = self.codebook_proj(condition)
        
        x = torch.cat([a_t, state, t_emb, c_emb], dim=-1)
        x = self.backbone(x)
        noise_pred = self.output(x)
        
        return noise_pred


class UnifiedNoisePredictor(nn.Module):
    """
    Unified noise predictor supporting both discrete and continuous policies.
    
    Uses mode flag to switch between ε_θd and ε_θc behavior.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        codebook_dim: Optional[int] = None,
        hidden_dim: int = 256,
        time_dim: int = 64,
        num_layers: int = 3,
        policy_type: str = "discrete"
    ):
        super().__init__()
        self.policy_type = policy_type
        
        if policy_type == "discrete":
            self.predictor = DiscreteNoisePredictor(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                num_layers=num_layers
            )
        elif policy_type == "continuous":
            assert codebook_dim is not None, "codebook_dim required for continuous policy"
            self.predictor = ContinuousNoisePredictor(
                state_dim=state_dim,
                action_dim=action_dim,
                codebook_dim=codebook_dim,
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                num_layers=num_layers
            )
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    
    def forward(
        self,
        a_t: torch.Tensor,
        state: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through noise predictor."""
        return self.predictor(a_t, state, t, condition)
