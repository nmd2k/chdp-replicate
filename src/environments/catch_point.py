"""Catch Point environment - Intercept moving target."""

from typing import Any, Dict, Tuple

import numpy as np
from gymnasium import spaces

from .base_pamdp import BasePAMDP


class CatchPointEnv(BasePAMDP):
    """
    Moving target interception environment.
    
    The agent must catch a moving target point. The target moves according
    to a predefined pattern (linear, circular, or random walk).
    
    State space: [agent_x, agent_y, target_x, target_y, target_vx, target_vy]
    Action space: (discrete: movement_mode, continuous: [direction_x, direction_y, speed])
    
    Movement modes:
        0: Move (standard movement)
        1: Predict (move to predicted intercept point)
        2: Sprint (fast burst toward target)
    
    Success: Catch the target (distance < catch_radius)
    Failure: Timeout
    """

    def __init__(
        self,
        max_episode_steps: int = 100,
        render_mode: str | None = None,
        arena_size: float = 10.0,
        catch_radius: float = 0.5,
        target_speed: float = 1.0,
        target_motion_type: str = "linear",
    ):
        """
        Initialize the Catch Point environment.
        
        Args:
            max_episode_steps: Maximum steps per episode
            render_mode: Render mode
            arena_size: Size of the arena
            catch_radius: Radius for successful catch
            target_speed: Speed of the target
            target_motion_type: Type of target motion ("linear", "circular", "random")
        """
        super().__init__(max_episode_steps, render_mode)
        
        self.arena_size = arena_size
        self.catch_radius = catch_radius
        self.target_speed = target_speed
        self.target_motion_type = target_motion_type
        
        # Movement mode parameters: (max_speed, accuracy_bonus)
        self._mode_params = [
            (1.5, 0.0),    # Move
            (1.0, 0.5),    # Predict (slower but better accuracy)
            (3.0, -0.2),   # Sprint (fast but less accurate)
        ]
        
        # State: [agent_x, agent_y, target_x, target_y, target_vx, target_vy]
        self._state_space = spaces.Box(
            low=np.array([-arena_size, -arena_size, -arena_size, -arena_size, -target_speed, -target_speed], dtype=np.float32),
            high=np.array([arena_size, arena_size, arena_size, arena_size, target_speed, target_speed], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Discrete action: movement mode
        self._num_discrete_actions = len(self._mode_params)
        
        # Continuous parameters: [direction_x, direction_y, speed_multiplier]
        self._max_param_dim = 3
        self._action_space = spaces.Tuple((
            spaces.Discrete(self._num_discrete_actions),
            spaces.Box(
                low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
        ))
        
        self._target_pos = None
        self._target_vel = None
        self._target_time = 0.0
        
    def _get_initial_state(self) -> np.ndarray:
        """Get initial state with random positions."""
        # Random agent position
        agent_pos = self.np_random.uniform(
            low=np.array([-self.arena_size/2, -self.arena_size/2]),
            high=np.array([self.arena_size/2, self.arena_size/2])
        )
        
        # Random target position (far from agent)
        while True:
            self._target_pos = self.np_random.uniform(
                low=np.array([-self.arena_size/2, -self.arena_size/2]),
                high=np.array([self.arena_size/2, self.arena_size/2])
            )
            dist = np.linalg.norm(self._target_pos - agent_pos)
            if dist > self.arena_size / 3:
                break
        
        # Initialize target velocity based on motion type
        self._target_time = 0.0
        self._init_target_velocity()
        
        return np.array([
            agent_pos[0], agent_pos[1],
            self._target_pos[0], self._target_pos[1],
            self._target_vel[0], self._target_vel[1]
        ], dtype=np.float32)
    
    def _init_target_velocity(self):
        """Initialize target velocity based on motion type."""
        if self.target_motion_type == "linear":
            # Random direction
            angle = self.np_random.uniform(0, 2 * np.pi)
            self._target_vel = np.array([
                np.cos(angle) * self.target_speed,
                np.sin(angle) * self.target_speed
            ])
        elif self.target_motion_type == "circular":
            # Tangential velocity for circular motion
            center = np.array([0.0, 0.0])
            radius = np.linalg.norm(self._target_pos - center)
            if radius < 1.0:
                radius = 1.0
            direction = self._target_pos - center
            direction = np.array([-direction[1], direction[0]])  # Perpendicular
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            self._target_vel = direction * self.target_speed
        else:  # random
            self._target_vel = np.array([0.0, 0.0])
    
    def _update_target(self, dt: float) -> np.ndarray:
        """Update target position based on motion type."""
        self._target_time += dt
        
        if self.target_motion_type == "linear":
            # Move with constant velocity, bounce off walls
            new_pos = self._target_pos + self._target_vel * dt
            
            # Bounce off walls
            for i in range(2):
                if new_pos[i] < -self.arena_size/2 or new_pos[i] > self.arena_size/2:
                    self._target_vel[i] *= -1
                    new_pos[i] = np.clip(new_pos[i], -self.arena_size/2, self.arena_size/2)
            
            self._target_pos = new_pos
            
        elif self.target_motion_type == "circular":
            # Circular motion around center
            center = np.array([0.0, 0.0])
            radius = np.linalg.norm(self._target_pos - center)
            if radius < 1.0:
                radius = 1.0
            angle = self._target_time * self.target_speed / radius
            self._target_pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle)
            ])
            # Update velocity (tangential)
            direction = np.array([-self._target_pos[1], self._target_pos[0]])
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            self._target_vel = direction * self.target_speed
            
        else:  # random walk
            noise = self.np_random.normal(0, 0.5, size=2)
            new_pos = self._target_pos + noise * dt
            self._target_pos = np.clip(new_pos, -self.arena_size/2, self.arena_size/2)
            self._target_vel = noise / np.linalg.norm(noise) * self.target_speed if np.linalg.norm(noise) > 0 else self._target_vel
        
        return self._target_pos
    
    def _apply_action(self, state: np.ndarray, action: Tuple[int, np.ndarray]) -> np.ndarray:
        """Apply movement action and update target."""
        agent_x, agent_y, _, _, target_vx, target_vy = state
        mode, params = action
        
        dt = 0.02  # Time step
        
        # Get mode parameters
        max_speed, accuracy_bonus = self._mode_params[mode]
        
        # Normalize direction
        direction = params[:2]
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            # Default: move toward target
            target_x, target_y = self._target_pos
            direction = np.array([target_x - agent_x, target_y - agent_y])
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            else:
                direction = np.array([1.0, 0.0])
        
        # Get speed
        speed = params[2] * max_speed
        
        # Add accuracy bonus/penalty
        if accuracy_bonus > 0:
            # Better accuracy - adjust direction toward predicted position
            predicted_target = self._target_pos + np.array([target_vx, target_vy]) * 0.5
            direction = direction * (1 - accuracy_bonus) + \
                       np.array([predicted_target[0] - agent_x, predicted_target[1] - agent_y]) * accuracy_bonus
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
        else:
            # Worse accuracy - add noise to direction
            noise = self.np_random.normal(0, abs(accuracy_bonus), size=2)
            direction = direction + noise
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
        
        # Update agent position
        new_x = agent_x + direction[0] * speed * dt
        new_y = agent_y + direction[1] * speed * dt
        
        # Clip to arena bounds
        new_x = np.clip(new_x, -self.arena_size/2, self.arena_size/2)
        new_y = np.clip(new_y, -self.arena_size/2, self.arena_size/2)
        
        # Update target
        self._update_target(dt)
        
        return np.array([
            new_x, new_y,
            self._target_pos[0], self._target_pos[1],
            self._target_vel[0], self._target_vel[1]
        ], dtype=np.float32)
    
    def _compute_reward(self, state: np.ndarray, action: Tuple[int, np.ndarray], next_state: np.ndarray) -> float:
        """Compute reward based on distance to target."""
        agent_x, agent_y, target_x, target_y, _, _ = next_state
        
        dist = np.sqrt((agent_x - target_x)**2 + (agent_y - target_y)**2)
        
        # Success reward
        if dist <= self.catch_radius:
            return 100.0
        
        # Dense reward: negative distance
        return -dist - 0.01  # Small step penalty
    
    def _check_success(self, state: np.ndarray) -> bool:
        """Check if agent caught the target."""
        agent_x, agent_y, target_x, target_y, _, _ = state
        dist = np.sqrt((agent_x - target_x)**2 + (agent_y - target_y)**2)
        return dist <= self.catch_radius
    
    def _check_failure(self, state: np.ndarray) -> bool:
        """Check for failure conditions."""
        return False
