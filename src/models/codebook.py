"""
Q-Guided Codebook for CHDP (Section 4.3)

Implements:
- Learnable codebook E_ζ ∈ ℝ^(K×d_e)
- Vector Quantization (Eq. 12)
- Q-function guided codebook alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch.distributions import Categorical


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer (Eq. 12).
    
    k = argmin_k ||e - e_k||²
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        self._ema_count = torch.zeros(num_embeddings)
        self._ema_weight = torch.zeros(num_embeddings, embedding_dim)
        self.use_ema = False
    
    def forward(
        self,
        e: torch.Tensor,
        use_q_guidance: bool = False,
        q_values: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vector quantization with optional Q-guidance.
        
        Args:
            e: Continuous encoding [batch, dim]
            use_q_guidance: Whether to use Q-function guidance
            q_values: Q-values for each codeword [batch, K]
            
        Returns:
            quantized: Quantized output [batch, dim]
            indices: Codeword indices [batch]
            commitment_loss: Commitment loss scalar
            codebook_usage: Fraction of codebook used
        """
        flat_input = e.view(-1, self.embedding_dim)
        
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_input, self.embedding.weight.T)
            + torch.sum(self.embedding.weight ** 2, dim=1)
        )
        
        if use_q_guidance and q_values is not None:
            q_guided_distances = distances - q_values
            encoding_indices = torch.argmin(q_guided_distances, dim=1)
        else:
            encoding_indices = torch.argmin(distances, dim=1)
        
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(e.shape)
        
        if self.training and self.use_ema:
            self._update_ema(e.detach(), encodings.detach())
        
        commitment_loss = F.mse_loss(quantized.detach(), e)
        
        codebook_usage = encodings.sum(0).max() / encodings.sum(0).sum()
        
        quantized = e + (quantized - e).detach()
        
        return quantized, encoding_indices, commitment_loss, codebook_usage
    
    def _update_ema(self, inputs: torch.Tensor, encodings: torch.Tensor):
        """Update EMA for codebook."""
        self._ema_count = self.decay * self._ema_count + (1 - self.decay) * encodings.sum(0)
        
        n = self._ema_count.sum()
        self._ema_count = (self._ema_count + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
        
        dw = torch.matmul(encodings.T, inputs)
        self._ema_weight = self.decay * self._ema_weight + (1 - self.decay) * dw
        self.embedding.weight.data = self._ema_weight / self._ema_count.unsqueeze(1)
    
    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Lookup embeddings by indices."""
        return self.embedding(indices)


class QGuidedCodebook(nn.Module):
    """
    Q-Guided Codebook (Section 4.3).
    
    Maintains learnable codebook E_ζ ∈ ℝ^(K×d_e) and aligns
    codewords with high Q-value actions.
    """
    
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        q_guidance_weight: float = 1.0
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.q_guidance_weight = q_guidance_weight
        
        self.vq = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )
        
        self.q_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(
        self,
        e: torch.Tensor,
        q_network: Optional[nn.Module] = None,
        state: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Q-guided vector quantization.
        
        Args:
            e: Continuous encoding [batch, dim]
            q_network: Q-network for computing Q-values
            state: State for Q-value computation
            training: Training mode flag
            
        Returns:
            quantized: Quantized encoding [batch, dim]
            indices: Codeword indices [batch]
            total_loss: Combined VQ + Q-guidance loss
            info: Additional information dict
        """
        q_values = None
        if q_network is not None and state is not None:
            with torch.no_grad():
                codewords = self.vq.embedding.weight.unsqueeze(0).expand(state.shape[0], -1, -1)
                state_expanded = state.unsqueeze(1).expand(-1, self.codebook_size, -1)
                
                q_values = q_network(state_expanded.reshape(-1, state.shape[-1] + self.embedding_dim))
                q_values = q_values.view(state.shape[0], self.codebook_size)
        
        use_q_guidance = training and q_values is not None
        
        quantized, indices, commitment_loss, codebook_usage = self.vq(
            e, use_q_guidance=use_q_guidance, q_values=q_values
        )
        
        vq_loss = commitment_loss * self.commitment_cost
        
        q_guidance_loss = torch.tensor(0.0, device=e.device)
        if q_values is not None:
            q_guidance_loss = -q_values.gather(1, indices.unsqueeze(1)).mean()
            q_guidance_loss = q_guidance_loss * self.q_guidance_weight
        
        total_loss = vq_loss + q_guidance_loss
        
        info = {
            'commitment_loss': commitment_loss.item(),
            'codebook_usage': codebook_usage.item(),
            'q_guidance_loss': q_guidance_loss.item() if isinstance(q_guidance_loss, torch.Tensor) else 0.0,
            'indices': indices
        }
        
        return quantized, indices, total_loss, info
    
    def encode(self, e: torch.Tensor) -> torch.Tensor:
        """Encode continuous vector to codeword index."""
        distances = (
            torch.sum(e ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(e, self.vq.embedding.weight.T)
            + torch.sum(self.vq.embedding.weight ** 2, dim=1)
        )
        return torch.argmin(distances, dim=1)
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codeword index to embedding."""
        return self.vq.lookup(indices)
    
    def get_codebook(self) -> torch.Tensor:
        """Get full codebook."""
        return self.vq.embedding.weight
