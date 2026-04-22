"""Platform environment - Agent navigates platforms with jumps."""

from typing import Any, Dict, Tuple

import numpy as np
from gymnasium import spaces

from .base_pamdp import BasePAMDP


class PlatformEnv(BasePAMDP):
    """
    Platform navigation environment.
    
    The agent must navigate across platforms by jumping. Each jump action
    has a discrete component (which platform to target) and continuous
    parameters (jump force/angle).
    
    State space: [agent_x, agent_y, agent_vx, agent_vy]
    Action space: (discrete: platform_id, continuous: [force_x, force_y])
    
    Success: Reach the goal platform
    Failure: Fall off the platforms
    """

    def __init__(
        self,
        max_episode_steps: int = 100,
        render_mode: str | None = None,
        platform_width: float = 2.0,
        gap_size: float = 1.0,
        num_platforms: int = 5,
        gravity: float = 9.8,
    ):
        """
        Initialize the Platform environment.
        
        Args:
            max_episode_steps: Maximum steps per episode
            render_mode: Render mode
            platform_width: Width of each platform
            gap_size: Gap between platforms
            num_platforms: Number of platforms
            gravity: Gravity constant
        """
        super().__init__(max_episode_steps, render_mode)
        
        self.platform_width = platform_width
        self.gap_size = gap_size
        self.num_platforms = num_platforms
        self.gravity = gravity
        self.dt = 0.02  # Time step
        
        # State: [x, y, vx, vy]
        self._state_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Discrete action: which platform to jump toward (0 to num_platforms-1)
        self._num_discrete_actions = num_platforms
        
        # Continuous parameters: [force_x, force_y]
        self._max_param_dim = 2
        self._action_space = spaces.Tuple((
            spaces.Discrete(self._num_discrete_actions),
            spaces.Box(
                low=np.array([-50.0, 0.0], dtype=np.float32),
                high=np.array([50.0, 50.0], dtype=np.float32),
                dtype=np.float32,
            ),
        ))
        
        # Platform positions
        self._platforms = []
        self._goal_platform = num_platforms - 1
        self._start_platform = 0
        
    def _generate_platforms(self):
        """Generate platform positions."""
        self._platforms = []
        for i in range(self.num_platforms):
            x_center = i * (self.platform_width + self.gap_size)
            x_left = x_center - self.platform_width / 2
            x_right = x_center + self.platform_width / 2
            self._platforms.append((x_left, x_right))
    
    def _get_initial_state(self) -> np.ndarray:
        """Get initial state on the starting platform."""
        self._generate_platforms()
        start_x = (self._platforms[0][0] + self._platforms[0][1]) / 2
        start_y = 0.0
        return np.array([start_x, start_y, 0.0, 0.0], dtype=np.float32)
    
    def _apply_action(self, state: np.ndarray, action: Tuple[int, np.ndarray]) -> np.ndarray:
        """Apply jump action and simulate physics."""
        x, y, vx, vy = state
        discrete_action, params = action
        
        # Apply impulse from jump
        force_x, force_y = params
        vx += force_x * self.dt
        vy += force_y * self.dt
        
        # Simulate physics for a short duration
        for _ in range(10):
            # Apply gravity
            vy -= self.gravity * self.dt
            
            # Update position
            x += vx * self.dt
            y += vy * self.dt
            
            # Check if on a platform
            on_platform = False
            for i, (x_left, x_right) in enumerate(self._platforms):
                if x_left <= x <= x_right and abs(y) < 0.1:
                    y = 0.0
                    vy = max(0, vy)  # Stop downward velocity
                    vx *= 0.9  # Friction
                    on_platform = True
                    break
            
            if not on_platform and y < -10:  # Fell too far
                break
        
        return np.array([x, y, vx, vy], dtype=np.float32)
    
    def _compute_reward(self, state: np.ndarray, action: Tuple[int, np.ndarray], next_state: np.ndarray) -> float:
        """Compute reward based on progress toward goal."""
        x, y, _, _ = next_state
        
        # Check if on goal platform
        goal_x_left, goal_x_right = self._platforms[self._goal_platform]
        if goal_x_left <= x <= goal_x_right and abs(y) < 0.1:
            return 100.0  # Success reward
        
        # Penalty for falling
        if y < -5:
            return -10.0
        
        # Reward for progress (distance toward goal)
        goal_center = (goal_x_left + goal_x_right) / 2
        start_center = (self._platforms[0][0] + self._platforms[0][1]) / 2
        total_dist = goal_center - start_center
        current_dist = x - start_center
        progress_reward = (current_dist / total_dist) * 1.0 if total_dist > 0 else 0.0
        
        return progress_reward - 0.01  # Small step penalty
    
    def _check_success(self, state: np.ndarray) -> bool:
        """Check if agent reached the goal platform."""
        x, y, _, _ = state
        goal_x_left, goal_x_right = self._platforms[self._goal_platform]
        return goal_x_left <= x <= goal_x_right and abs(y) < 0.1
    
    def _check_failure(self, state: np.ndarray) -> bool:
        """Check if agent fell off the platforms."""
        _, y, _, _ = state
        return y < -5.0
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            super().render()
            if hasattr(self, '_state'):
                x, y, _, _ = self._state
                print(f"Position: ({x:.2f}, {y:.2f})")
                print(f"Goal platform: {self._goal_platform}")
        return None
