"""Base class for Parameterized Action Markov Decision Processes (PAMDP)."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BasePAMDP(gym.Env, ABC):
    """
    Base class for Parameterized Action Markov Decision Processes.
    
    PAMDPs have hybrid action spaces consisting of:
    - A discrete action (selecting which parameterized skill to execute)
    - Continuous parameters for the selected skill
    
    The action space is a tuple: (discrete_action, continuous_parameters)
    where the dimensionality of continuous_parameters may depend on the discrete action.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        max_episode_steps: int = 100,
        render_mode: str | None = None,
    ):
        """
        Initialize the PAMDP environment.
        
        Args:
            max_episode_steps: Maximum number of steps per episode
            render_mode: Render mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._current_step = 0
        self._episode_success = False
        self._episode_failure = False
        
        # These must be set by subclasses
        self._state_space: spaces.Space | None = None
        self._action_space: spaces.Tuple | None = None
        self._num_discrete_actions: int = 0
        self._max_param_dim: int = 0  # Maximum continuous parameter dimension
        
    @property
    def observation_space(self) -> spaces.Space:
        """Return the observation space."""
        return self._state_space
        
    @property
    def state_space(self) -> spaces.Space:
        """Return the state space (alias for observation_space)."""
        return self._state_space
        
    @property
    def action_space(self) -> spaces.Tuple:
        """Return the hybrid action space (discrete, continuous)."""
        return self._action_space
        
    @property
    def num_discrete_actions(self) -> int:
        """Return the number of discrete actions."""
        return self._num_discrete_actions
        
    @property
    def max_param_dim(self) -> int:
        """Return the maximum continuous parameter dimension."""
        return self._max_param_dim

    @abstractmethod
    def _get_initial_state(self) -> np.ndarray:
        """
        Get the initial state for a new episode.
        
        Returns:
            Initial state as numpy array
        """
        pass
        
    @abstractmethod
    def _compute_reward(self, state: np.ndarray, action: Tuple[int, np.ndarray], next_state: np.ndarray) -> float:
        """
        Compute the reward for a transition.
        
        Args:
            state: Current state
            action: Tuple of (discrete_action, continuous_params)
            next_state: Next state after action
            
        Returns:
            Reward value
        """
        pass
        
    @abstractmethod
    def _check_success(self, state: np.ndarray) -> bool:
        """
        Check if the current state indicates success.
        
        Args:
            state: Current state
            
        Returns:
            True if success condition is met
        """
        pass
        
    @abstractmethod
    def _check_failure(self, state: np.ndarray) -> bool:
        """
        Check if the current state indicates failure.
        
        Args:
            state: Current state
            
        Returns:
            True if failure condition is met
        """
        pass
        
    @abstractmethod
    def _apply_action(self, state: np.ndarray, action: Tuple[int, np.ndarray]) -> np.ndarray:
        """
        Apply an action to the current state and compute the next state.
        
        Args:
            state: Current state
            action: Tuple of (discrete_action, continuous_params)
            
        Returns:
            Next state
        """
        pass

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (initial_state, info_dict)
        """
        super().reset(seed=seed)
        
        self._current_step = 0
        self._episode_success = False
        self._episode_failure = False
        
        state = self._get_initial_state()
        info = {"success": False, "failure": False}
        
        return state, info

    def step(
        self,
        action: Tuple[int, np.ndarray | list],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Tuple of (discrete_action, continuous_params)
                - discrete_action: int in [0, num_discrete_actions)
                - continuous_params: array of parameters for the action
                
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        discrete_action = int(action[0])
        continuous_params = np.asarray(action[1], dtype=np.float32)
        action_tuple = (discrete_action, continuous_params)
        
        # Apply action
        current_state = self._state if hasattr(self, '_state') else self._get_initial_state()
        next_state = self._apply_action(current_state, action_tuple)
        
        # Compute reward
        reward = self._compute_reward(current_state, action_tuple, next_state)
        
        # Check termination conditions
        self._episode_success = self._check_success(next_state)
        self._episode_failure = self._check_failure(next_state)
        terminated = self._episode_success or self._episode_failure
        
        # Check truncation
        self._current_step += 1
        truncated = self._current_step >= self.max_episode_steps
        
        # Update state
        self._state = next_state
        
        # Info dict
        info = {
            "success": self._episode_success,
            "failure": self._episode_failure,
            "episode_success": self._episode_success,
        }
        
        return next_state, reward, terminated, truncated, info

    @property
    def success_rate(self) -> float:
        """Return 1.0 if episode was successful, 0.0 otherwise."""
        return 1.0 if self._episode_success else 0.0

    def render(self):
        """Render the environment (not implemented by default)."""
        if self.render_mode == "human":
            print(f"Step: {self._current_step}/{self.max_episode_steps}")
            if hasattr(self, '_state'):
                print(f"State: {self._state}")
        return None

    def close(self):
        """Clean up resources."""
        pass

    def get_param_dim(self, discrete_action: int) -> int:
        """
        Get the parameter dimension for a specific discrete action.
        
        Args:
            discrete_action: The discrete action index
            
        Returns:
            Parameter dimension for that action
        """
        # Default: all actions have the same parameter dimension
        return self._max_param_dim
