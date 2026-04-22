"""CHDP Agent with sequential update scheme."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from copy import deepcopy


class VectorQuantizer(nn.Module):
    """
    Vector quantization layer for discrete latent codes.
    
    Maps continuous embeddings to discrete codebook entries.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        commitment_loss_weight: float = 0.25
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_loss_weight = commitment_loss_weight
        
        # Codebook: K x D
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(
        self,
        z: torch.Tensor,
        return_indices: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Quantize continuous input to nearest codebook entry.
        
        Args:
            z: Continuous embeddings of shape (B, D)
            return_indices: Whether to return quantization indices
            
        Returns:
            z_q: Quantized embeddings (B, D)
            vq_loss: Vector quantization loss
            indices: Quantization indices (B,) if return_indices=True
        """
        # Flatten input
        z_flattened = z.view(-1, self.embedding_dim)
        
        # Compute distances to codebook entries
        # (B, D) -> (B, 1, D) - (K, D) -> (B, K)
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_flattened, self.codebook.weight.t())
            + torch.sum(self.codebook.weight ** 2, dim=1)
        )
        
        # Get nearest neighbors
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        z_q = torch.matmul(encodings, self.codebook.weight).view(z.shape)
        
        # Losses
        # Commitment loss: encourage encoder to commit to codebook
        commitment_loss = F.mse_loss(z_q.detach(), z)
        
        # Codebook loss: move codebook towards encoder output
        codebook_loss = F.mse_loss(z_q, z.detach())
        
        vq_loss = codebook_loss + self.commitment_loss_weight * commitment_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        if return_indices:
            return z_q, vq_loss, encoding_indices
        return z_q, vq_loss, None


class DiffusionPolicy(nn.Module):
    """
    Diffusion-based policy network.
    
    Can be used for both discrete and continuous policy learning.
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_diffusion_steps: int = 10,
        policy_type: str = "discrete"
    ):
        super().__init__()
        self.policy_type = policy_type
        self.num_diffusion_steps = num_diffusion_steps
        self.action_dim = action_dim
        
        # Input processing
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Time embedding for diffusion
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Denoising network
        if policy_type == "discrete":
            # Output logits for discrete actions
            self.denoise_net = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        else:
            # Output continuous action noise prediction
            self.denoise_net = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        # Diffusion schedule
        self.register_buffer(
            "betas",
            torch.linspace(1e-4, 0.02, num_diffusion_steps)
        )
        self._precompute_diffusion_vars()
    
    def _precompute_diffusion_vars(self) -> None:
        """Precompute diffusion schedule variables."""
        alphas = 1.0 - self.betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
    
    def forward(
        self,
        state: torch.Tensor,
        latent: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        time_step: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through diffusion policy.
        
        Args:
            state: Current state (B, state_dim)
            latent: Latent code (B, latent_dim)
            action: Action to denoise (B, action_dim) or None for sampling
            time_step: Diffusion time step for training
            training: Whether in training mode
            
        Returns:
            Predicted action or noise
        """
        # Encode inputs
        state_feat = self.state_encoder(state)
        latent_feat = self.latent_encoder(latent)
        
        if training and action is not None and time_step is not None:
            # Training: add noise and predict
            noise = torch.randn_like(action)
            alpha_bar_t = self.alpha_bar[time_step].view(-1, 1)
            
            # q(x_t | x_0, t)
            noisy_action = torch.sqrt(alpha_bar_t) * action + torch.sqrt(1 - alpha_bar_t) * noise
            
            # Time embedding
            time_feat = self.time_embed(
                torch.sin(time_step.float().unsqueeze(-1) * torch.pi / self.num_diffusion_steps)
            )
            
            # Concatenate and denoise
            x = torch.cat([state_feat, latent_feat, time_feat, noisy_action], dim=-1)
            return self.denoise_net(x)
        else:
            # Sampling: reverse diffusion
            return self._sample(state_feat, latent_feat)
    
    def _sample(
        self,
        state_feat: torch.Tensor,
        latent_feat: torch.Tensor
    ) -> torch.Tensor:
        """Sample action using reverse diffusion."""
        batch_size = state_feat.shape[0]
        action = torch.randn(batch_size, self.action_dim, device=state_feat.device)
        
        for t in reversed(range(self.num_diffusion_steps)):
            t_tensor = torch.full((batch_size,), t, device=state_feat.device, dtype=torch.long)
            time_feat = self.time_embed(
                torch.sin(t_tensor.float().unsqueeze(-1) * torch.pi / self.num_diffusion_steps)
            )
            
            x = torch.cat([state_feat, latent_feat, time_feat, action], dim=-1)
            noise_pred = self.denoise_net(x)
            
            # Reverse diffusion step
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_prev = self.alpha_bar[t - 1] if t > 0 else torch.ones(1, device=action.device)
            
            beta_t = self.betas[t]
            
            # Mean calculation
            mean = (1.0 / torch.sqrt(alpha_t)) * (action - beta_t / torch.sqrt(1 - alpha_bar_t) * noise_pred)
            
            # Variance
            variance = beta_t if t > 0 else 0
            
            # Sample
            if t > 0:
                action = mean + torch.sqrt(variance) * torch.randn_like(mean)
            else:
                action = mean
        
        return action


class Critic(nn.Module):
    """
    Q-value critic network.
    
    Estimates Q(s, e, a^c) for CHDP training.
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.q_net = nn.Sequential(
            nn.Linear(state_dim + latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        state: torch.Tensor,
        latent: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-value for state-latent-action tuple.
        
        Args:
            state: State (B, state_dim)
            latent: Latent code (B, latent_dim)
            action: Continuous action (B, action_dim)
            
        Returns:
            Q-value (B, 1)
        """
        x = torch.cat([state, latent, action], dim=-1)
        return self.q_net(x)


class CHDPAgent:
    """
    CHDP Agent with sequential update scheme.
    
    Implements the two-stage update:
    1. Update discrete policy (θ_d)
    2. Update continuous policy (θ_c) and codebook (ζ)
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        action_dim: int,
        num_codes: int = 64,
        hidden_dim: int = 256,
        lr_discrete: float = 3e-4,
        lr_continuous: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_codebook: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 0.1,
        tau: float = 0.005,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        
        # Discrete policy
        self.policy_discrete = DiffusionPolicy(
            state_dim, latent_dim, num_codes,
            hidden_dim, policy_type="discrete"
        ).to(device)
        
        # Continuous policy
        self.policy_continuous = DiffusionPolicy(
            state_dim, latent_dim, action_dim,
            hidden_dim, policy_type="continuous"
        ).to(device)
        
        # Vector quantizer (codebook)
        self.quantizer = VectorQuantizer(latent_dim, num_codes).to(device)
        
        # Double critics
        self.critic_1 = Critic(state_dim, latent_dim, action_dim, hidden_dim).to(device)
        self.critic_2 = Critic(state_dim, latent_dim, action_dim, hidden_dim).to(device)
        
        # Target critics (for stability)
        self.critic_target_1 = deepcopy(self.critic_1)
        self.critic_target_2 = deepcopy(self.critic_2)
        
        # Optimizers
        self.optimizer_discrete = torch.optim.Adam(
            self.policy_discrete.parameters(), lr=lr_discrete
        )
        self.optimizer_continuous = torch.optim.Adam(
            self.policy_continuous.parameters(), lr=lr_continuous
        )
        self.optimizer_critic = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=lr_critic
        )
        self.optimizer_codebook = torch.optim.Adam(
            self.quantizer.parameters(), lr=lr_codebook
        )
    
    def select_action(
        self,
        state: np.ndarray,
        latent: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select action given state.
        
        Args:
            state: Current state
            latent: Optional latent code (if None, will be sampled)
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, latent_code)
        """
        self.policy_discrete.eval()
        self.policy_continuous.eval()
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Sample or use provided latent
            if latent is None:
                # Sample discrete latent from discrete policy
                latent_t = self._sample_discrete_latent(state_t)
            else:
                latent_t = torch.FloatTensor(latent).unsqueeze(0).to(self.device)
            
            # Get continuous action
            action_t = self.policy_continuous(
                state_t, latent_t, training=False
            )
            
            if deterministic:
                action = action_t.cpu().numpy()[0]
            else:
                # Add exploration noise
                action = action_t.cpu().numpy()[0] + np.random.normal(0, 0.1, action_t.shape[1])
            
            latent_out = latent_t.cpu().numpy()[0]
        
        self.policy_discrete.train()
        self.policy_continuous.train()
        
        return action, latent_out
    
    def _sample_discrete_latent(self, state: torch.Tensor) -> torch.Tensor:
        """Sample discrete latent code from discrete policy."""
        # Get logits from discrete policy
        latent_feat = self.policy_discrete.latent_encoder(
            self.policy_discrete.state_encoder(state)
        )
        
        # For simplicity, sample uniformly (can be enhanced with Gumbel-softmax)
        batch_size = state.shape[0]
        indices = torch.randint(
            0, self.quantizer.num_embeddings,
            (batch_size,), device=self.device
        )
        
        # Get codebook embedding
        latent = self.quantizer.codebook(indices)
        return latent
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform sequential update of CHDP agent.
        
        Step 1: Update discrete policy (θ_d)
        Step 2: Update continuous policy (θ_c) and codebook (ζ)
        Step 3: Update critics (φ)
        
        Args:
            batch: Dictionary with keys: states, latents, actions, rewards, next_states, dones
            
        Returns:
            Dictionary of loss values
        """
        states = batch["states"]
        latents = batch["latents"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        
        losses = {}
        
        # =========================================================================
        # Step 1: Discrete Policy Update (Eq. 7)
        # L(θ_d) = L_d(θ_d) - α * E[Q_φ(s, e, a^c)]
        # =========================================================================
        
        self.optimizer_discrete.zero_grad()
        
        # Get discrete policy output
        discrete_logits = self.policy_discrete(states, latents, training=True)
        
        # Diffusion loss for discrete policy
        # (simplified: MSE between predicted and target)
        discrete_loss = F.mse_loss(discrete_logits, torch.zeros_like(discrete_logits))
        
        # Q-value term: negative because we want to maximize Q
        # Use fixed latents/actions from replay buffer (a^c is the target)
        q_discrete_1 = self.critic_1(states, latents, actions)
        q_discrete_2 = self.critic_2(states, latents, actions)
        q_discrete = torch.min(q_discrete_1, q_discrete_2)
        
        # Total discrete loss
        loss_discrete = discrete_loss - self.alpha * q_discrete.mean()
        
        loss_discrete.backward()
        self.optimizer_discrete.step()
        
        losses["loss_discrete"] = loss_discrete.item()
        
        # =========================================================================
        # Step 2: Continuous Policy + Codebook Update (Eq. 8-9)
        # L(θ_c, ζ) = L_d(θ_c) + α * L_q(θ_c, ζ)
        # L_q(θ_c, ζ) = -E[Q_φ(s, sg(e'), a^c)]
        # =========================================================================
        
        self.optimizer_continuous.zero_grad()
        self.optimizer_codebook.zero_grad()
        
        # Generate new latents from continuous policy pathway
        # First, get continuous representation
        continuous_feat = self.policy_continuous.state_encoder(states)
        
        # Quantize to get discrete codes
        latents_quantized, vq_loss, indices = self.quantizer(
            continuous_feat, return_indices=True
        )
        
        # Stop-gradient: prevent gradients from flowing back to discrete policy
        # This is key to the sequential update scheme
        latents_sg = latents_quantized.detach()
        
        # Continuous policy loss
        continuous_output = self.policy_continuous(
            states, latents_sg, actions, training=True
        )
        continuous_loss = F.mse_loss(continuous_output, actions)
        
        # Q-value loss with stop-gradient on latents
        # The sg(e') means we don't update discrete policy in this step
        q_continuous_1 = self.critic_1(states, latents_sg, actions)
        q_continuous_2 = self.critic_2(states, latents_sg, actions)
        q_continuous = torch.min(q_continuous_1, q_continuous_2)
        
        # Negative Q because we want to maximize
        q_loss_continuous = -self.alpha * q_continuous.mean()
        
        # Total loss for continuous policy and codebook
        loss_continuous = continuous_loss + q_loss_continuous + vq_loss
        
        loss_continuous.backward()
        self.optimizer_continuous.step()
        self.optimizer_codebook.step()
        
        losses["loss_continuous"] = loss_continuous.item()
        losses["vq_loss"] = vq_loss.item()
        
        # =========================================================================
        # Step 3: Critic Update (Eq. 10-11)
        # L(φ_i) = E[(Q_φ_i(s, e, a^c) - y_t)²]
        # y_t = r_t + γ * min_j Q'_φ'_j(s_{t+1}, e_{t+1}, a^c_{t+1})
        # =========================================================================
        
        self.optimizer_critic.zero_grad()
        
        # Compute target Q-values
        with torch.no_grad():
            # Get next state latents (with stop-gradient)
            next_continuous_feat = self.policy_continuous.state_encoder(next_states)
            next_latents, _, _ = self.quantizer(next_continuous_feat, return_indices=True)
            next_latents = next_latents.detach()
            
            # Sample next actions
            next_actions = self.policy_continuous(
                next_states, next_latents, training=False
            )
            
            # Target Q-values from both critics (double Q-learning)
            next_q_1 = self.critic_target_1(next_states, next_latents, next_actions)
            next_q_2 = self.critic_target_2(next_states, next_latents, next_actions)
            next_q_min = torch.min(next_q_1, next_q_2)
            
            # Bellman backup
            targets = rewards + self.gamma * (1 - dones) * next_q_min
        
        # Compute current Q-values
        current_q_1 = self.critic_1(states, latents, actions)
        current_q_2 = self.critic_2(states, latents, actions)
        
        # Critic losses
        critic_loss_1 = F.mse_loss(current_q_1, targets)
        critic_loss_2 = F.mse_loss(current_q_2, targets)
        critic_loss = critic_loss_1 + critic_loss_2
        
        critic_loss.backward()
        self.optimizer_critic.step()
        
        losses["critic_loss"] = critic_loss.item()
        
        # =========================================================================
        # Soft update of target networks
        # =========================================================================
        self._soft_update_targets()
        
        return losses
    
    def _soft_update_targets(self) -> None:
        """Soft update of target critic networks."""
        for param, target_param in zip(
            self.critic_1.parameters(), self.critic_target_1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        for param, target_param in zip(
            self.critic_2.parameters(), self.critic_target_2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str) -> None:
        """Save agent checkpoints."""
        torch.save({
            "policy_discrete": self.policy_discrete.state_dict(),
            "policy_continuous": self.policy_continuous.state_dict(),
            "quantizer": self.quantizer.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_target_1": self.critic_target_1.state_dict(),
            "critic_target_2": self.critic_target_2.state_dict(),
            "optimizer_discrete": self.optimizer_discrete.state_dict(),
            "optimizer_continuous": self.optimizer_continuous.state_dict(),
            "optimizer_codebook": self.optimizer_codebook.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_discrete.load_state_dict(checkpoint["policy_discrete"])
        self.policy_continuous.load_state_dict(checkpoint["policy_continuous"])
        self.quantizer.load_state_dict(checkpoint["quantizer"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.critic_target_1.load_state_dict(checkpoint["critic_target_1"])
        self.critic_target_2.load_state_dict(checkpoint["critic_target_2"])
        self.optimizer_discrete.load_state_dict(checkpoint["optimizer_discrete"])
        self.optimizer_continuous.load_state_dict(checkpoint["optimizer_continuous"])
        self.optimizer_codebook.load_state_dict(checkpoint["optimizer_codebook"])
        self.optimizer_critic.load_state_dict(checkpoint["optimizer_critic"])
