"""Hard Move environment - Combinatorial actuator control."""

from itertools import product
from typing import Any, Dict, Tuple

import numpy as np
from gymnasium import spaces

from .base_pamdp import BasePAMDP


class HardMoveEnv(BasePAMDP):
    """
    Combinatorial actuator control environment.
    
    The agent controls n actuators, each can be ON or OFF.
    This creates 2^n discrete actions (combinatorial explosion).
    Each actuator has a continuous parameter (force/intensity).
    
    The goal is to reach a target configuration of actuator states
    while minimizing energy usage.
    
    State space: [actuator_1_state, ..., actuator_n_state, target_1, ..., target_n]
    Action space: (discrete: actuator_configuration, continuous: [force_1, ..., force_n])
    
    Args:
        n: Number of actuators (creates 2^n discrete actions)
    """

    def __init__(
        self,
        n: int = 4,
        max_episode_steps: int = 100,
        render_mode: str | None = None,
        actuator_range: float = 1.0,
        target_noise: float = 0.1,
    ):
        """
        Initialize the Hard Move environment.
        
        Args:
            n: Number of actuators (default 4, creates 16 discrete actions)
            max_episode_steps: Maximum steps per episode
            render_mode: Render mode
            actuator_range: Range of actuator states [-range, range]
            target_noise: Noise in target configuration
        """
        super().__init__(max_episode_steps, render_mode)
        
        self.n = n
        self.actuator_range = actuator_range
        self.target_noise = target_noise
        
        # Generate all 2^n configurations
        self._configurations = list(product([0, 1], repeat=n))
        
        # State: [actuator_states (n), target_states (n)]
        self._state_space = spaces.Box(
            low=np.array([-actuator_range] * n + [-actuator_range] * n, dtype=np.float32),
            high=np.array([actuator_range] * n + [actuator_range] * n, dtype=np.float32),
            dtype=np.float32,
        )
        
        # Discrete action: which configuration to activate (2^n options)
        self._num_discrete_actions = len(self._configurations)
        
        # Continuous parameters: force for each actuator
        self._max_param_dim = n
        self._action_space = spaces.Tuple((
            spaces.Discrete(self._num_discrete_actions),
            spaces.Box(
                low=np.array([-1.0] * n, dtype=np.float32),
                high=np.array([1.0] * n, dtype=np.float32),
                dtype=np.float32,
            ),
        ))
        
        self._target_states = None
        
    def _get_initial_state(self) -> np.ndarray:
        """Get initial state with random actuator and target states."""
        # Random initial actuator states
        actuator_states = self.np_random.uniform(
            low=-self.actuator_range,
            high=self.actuator_range,
            size=self.n
        )
        
        # Random target states
        self._target_states = self.np_random.uniform(
            low=-self.actuator_range + self.target_noise,
            high=self.actuator_range - self.target_noise,
            size=self.n
        )
        
        return np.concatenate([actuator_states, self._target_states]).astype(np.float32)
    
    def _apply_action(self, state: np.ndarray, action: Tuple[int, np.ndarray]) -> np.ndarray:
        """Apply actuator configuration and forces."""
        actuator_states = state[:self.n].copy()
        target_states = state[self.n:].copy()
        
        config_idx, forces = action
        
        # Get the configuration (which actuators are ON)
        config = self._configurations[config_idx]
        
        # Update actuator states based on configuration and forces
        dt = 0.05  # Time step
        for i in range(self.n):
            if config[i] == 1:  # Actuator is ON
                # Apply force
                actuator_states[i] += forces[i] * dt
                # Add some dynamics (spring-like return to center)
                actuator_states[i] -= 0.1 * actuator_states[i] * dt
            else:  # Actuator is OFF
                # Decay toward zero
                actuator_states[i] *= (1 - 0.2 * dt)
            
            # Clip to valid range
            actuator_states[i] = np.clip(actuator_states[i], -self.actuator_range, self.actuator_range)
        
        return np.concatenate([actuator_states, target_states]).astype(np.float32)
    
    def _compute_reward(self, state: np.ndarray, action: Tuple[int, np.ndarray], next_state: np.ndarray) -> float:
        """Compute reward based on distance to target and energy usage."""
        actuator_states = next_state[:self.n]
        target_states = next_state[self.n:]
        config_idx, forces = action
        
        # Distance to target
        dist = np.linalg.norm(actuator_states - target_states)
        
        # Energy penalty (based on forces and active actuators)
        config = self._configurations[config_idx]
        energy = np.sum(np.abs(forces) * np.array(config))
        
        # Success reward
        if dist < 0.1 * self.actuator_range:
            return 100.0 - 0.1 * energy
        
        # Dense reward: negative distance and energy
        return -dist - 0.01 * energy - 0.001
    
    def _check_success(self, state: np.ndarray) -> bool:
        """Check if actuators reached target configuration."""
        actuator_states = state[:self.n]
        target_states = state[self.n:]
        dist = np.linalg.norm(actuator_states - target_states)
        return dist < 0.1 * self.actuator_range
    
    def _check_failure(self, state: np.ndarray) -> bool:
        """Check for failure conditions."""
        return False
    
    def get_param_dim(self, discrete_action: int) -> int:
        """Return parameter dimension (equals number of actuators)."""
        return self.n
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            super().render()
            if hasattr(self, '_state'):
                actuator_states = self._state[:self.n]
                target_states = self._state[self.n:]
                print(f"Actuators: {actuator_states}")
                print(f"Targets:   {target_states}")
                print(f"Configs:   {self._num_discrete_actions} (2^{self.n})")
        return None
