"""Replay buffer for CHDP agent training."""

import numpy as np
import torch
from collections import deque
from typing import Dict, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class Transition:
    """Single transition tuple (s, e, a^c, r, s', done)."""
    state: np.ndarray
    latent: np.ndarray  # e - discrete latent code
    action: np.ndarray  # a^c - continuous action
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Replay buffer for storing and sampling transitions.
    
    Stores (s, e, a^c, r, s') tuples for training the CHDP agent.
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        latent_dim: int,
        action_dim: int,
        device: str = "cpu"
    ):
        self.capacity = capacity
        self.device = torch.device(device)
        
        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.latents = np.zeros((capacity, latent_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.size = 0
        self.ptr = 0
    
    def add(
        self,
        state: np.ndarray,
        latent: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a transition to the buffer."""
        self.states[self.ptr] = state
        self.latents[self.ptr] = latent
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions.
        
        Returns:
            Dictionary with keys: states, latents, actions, rewards, next_states, dones
        """
        indices = np.random.randint(0, self.size, batch_size)
        
        batch = {
            "states": torch.FloatTensor(self.states[indices]).to(self.device),
            "latents": torch.FloatTensor(self.latents[indices]).to(self.device),
            "actions": torch.FloatTensor(self.actions[indices]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[indices]).to(self.device),
            "next_states": torch.FloatTensor(self.next_states[indices]).to(self.device),
            "dones": torch.FloatTensor(self.dones[indices]).to(self.device),
        }
        
        return batch
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size
