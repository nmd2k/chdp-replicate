"""Goal and Hard Goal environments - Reach target position with movement modes."""

from typing import Any, Dict, Tuple

import numpy as np
from gymnasium import spaces

from .base_pamdp import BasePAMDP


class GoalEnv(BasePAMDP):
    """
    Goal-reaching environment.
    
    The agent must reach a target position using different movement modes.
    Each movement mode has different characteristics (speed, precision).
    
    State space: [agent_x, agent_y, goal_x, goal_y]
    Action space: (discrete: movement_mode, continuous: [direction_x, direction_y, magnitude])
    
    Movement modes:
        0: Walk (slow, precise)
        1: Run (fast, less precise)
        2: Dash (very fast, imprecise)
    
    Success: Reach within goal radius
    Failure: Timeout (handled by truncation)
    """

    def __init__(
        self,
        max_episode_steps: int = 100,
        render_mode: str | None = None,
        arena_size: float = 10.0,
        goal_radius: float = 0.5,
        start_noise: float = 1.0,
    ):
        """
        Initialize the Goal environment.
        
        Args:
            max_episode_steps: Maximum steps per episode
            render_mode: Render mode
            arena_size: Size of the arena
            goal_radius: Radius of goal area
            start_noise: Noise in starting position
        """
        super().__init__(max_episode_steps, render_mode)
        
        self.arena_size = arena_size
        self.goal_radius = goal_radius
        self.start_noise = start_noise
        
        # Movement mode parameters: (max_speed, noise_std)
        self._mode_params = [
            (1.0, 0.05),   # Walk
            (2.0, 0.15),   # Run
            (4.0, 0.3),    # Dash
        ]
        
        # State: [agent_x, agent_y, goal_x, goal_y]
        self._state_space = spaces.Box(
            low=np.array([-arena_size, -arena_size, -arena_size, -arena_size], dtype=np.float32),
            high=np.array([arena_size, arena_size, arena_size, arena_size], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Discrete action: movement mode
        self._num_discrete_actions = len(self._mode_params)
        
        # Continuous parameters: [direction_x, direction_y, magnitude]
        self._max_param_dim = 3
        self._action_space = spaces.Tuple((
            spaces.Discrete(self._num_discrete_actions),
            spaces.Box(
                low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
        ))
        
        self._goal_pos = None
        self._start_pos = None
        
    def _get_initial_state(self) -> np.ndarray:
        """Get initial state with random start and goal positions."""
        # Random start position
        self._start_pos = self.np_random.uniform(
            low=np.array([-self.arena_size/2, -self.arena_size/2]),
            high=np.array([self.arena_size/2, self.arena_size/2])
        )
        
        # Random goal position (far from start)
        while True:
            self._goal_pos = self.np_random.uniform(
                low=np.array([-self.arena_size/2, -self.arena_size/2]),
                high=np.array([self.arena_size/2, self.arena_size/2])
            )
            dist = np.linalg.norm(self._goal_pos - self._start_pos)
            if dist > self.arena_size / 3:
                break
        
        return np.array([
            self._start_pos[0], self._start_pos[1],
            self._goal_pos[0], self._goal_pos[1]
        ], dtype=np.float32)
    
    def _apply_action(self, state: np.ndarray, action: Tuple[int, np.ndarray]) -> np.ndarray:
        """Apply movement action."""
        agent_x, agent_y, goal_x, goal_y = state
        mode, params = action
        
        # Get mode parameters
        max_speed, noise_std = self._mode_params[mode]
        
        # Normalize direction
        direction = params[:2]
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            direction = np.array([1.0, 0.0])
        
        # Get magnitude
        magnitude = params[2] * max_speed
        
        # Add noise based on mode
        noise = self.np_random.normal(0, noise_std, size=2)
        
        # Update position
        new_x = agent_x + (direction[0] * magnitude + noise[0])
        new_y = agent_y + (direction[1] * magnitude + noise[1])
        
        # Clip to arena bounds
        new_x = np.clip(new_x, -self.arena_size/2, self.arena_size/2)
        new_y = np.clip(new_y, -self.arena_size/2, self.arena_size/2)
        
        return np.array([new_x, new_y, goal_x, goal_y], dtype=np.float32)
    
    def _compute_reward(self, state: np.ndarray, action: Tuple[int, np.ndarray], next_state: np.ndarray) -> float:
        """Compute reward based on distance to goal."""
        agent_x, agent_y, goal_x, goal_y = next_state
        
        dist = np.sqrt((agent_x - goal_x)**2 + (agent_y - goal_y)**2)
        
        # Success reward
        if dist <= self.goal_radius:
            return 100.0
        
        # Dense reward: negative distance
        return -dist - 0.01  # Small step penalty
    
    def _check_success(self, state: np.ndarray) -> bool:
        """Check if agent reached the goal."""
        agent_x, agent_y, goal_x, goal_y = state
        dist = np.sqrt((agent_x - goal_x)**2 + (agent_y - goal_y)**2)
        return dist <= self.goal_radius
    
    def _check_failure(self, state: np.ndarray) -> bool:
        """Check for failure conditions."""
        return False  # No failure condition besides timeout


class HardGoalEnv(GoalEnv):
    """
    Harder goal-reaching environment.
    
    More challenging version with:
    - Smaller goal radius
    - More movement modes with trade-offs
    - Obstacles (implicit through penalty)
    - Higher noise in fast modes
    
    State space: [agent_x, agent_y, goal_x, goal_y, obstacle_x, obstacle_y]
    Action space: (discrete: movement_mode, continuous: [direction_x, direction_y, magnitude])
    
    Movement modes:
        0: Crawl (very slow, very precise)
        1: Walk (slow, precise)
        2: Run (fast, less precise)
        3: Dash (very fast, imprecise)
        4: Jump (fast, can skip obstacles)
    """

    def __init__(
        self,
        max_episode_steps: int = 100,
        render_mode: str | None = None,
        arena_size: float = 10.0,
        goal_radius: float = 0.3,
        start_noise: float = 1.0,
        num_obstacles: int = 3,
        obstacle_radius: float = 0.5,
    ):
        """
        Initialize the Hard Goal environment.
        
        Args:
            max_episode_steps: Maximum steps per episode
            render_mode: Render mode
            arena_size: Size of the arena
            goal_radius: Radius of goal area (smaller than easy version)
            start_noise: Noise in starting position
            num_obstacles: Number of obstacles
            obstacle_radius: Radius of each obstacle
        """
        super().__init__(max_episode_steps, render_mode, arena_size, goal_radius, start_noise)
        
        self.num_obstacles = num_obstacles
        self.obstacle_radius = obstacle_radius
        
        # More movement modes with different trade-offs
        self._mode_params = [
            (0.5, 0.02),   # Crawl
            (1.0, 0.05),   # Walk
            (2.5, 0.15),   # Run
            (5.0, 0.4),    # Dash
            (3.0, 0.2),    # Jump (can pass over obstacles)
        ]
        
        # State: [agent_x, agent_y, goal_x, goal_y] + obstacles
        self._num_state_vars = 4 + num_obstacles * 2
        self._state_space = spaces.Box(
            low=np.array([-arena_size] * self._num_state_vars, dtype=np.float32),
            high=np.array([arena_size] * self._num_state_vars, dtype=np.float32),
            dtype=np.float32,
        )
        
        self._num_discrete_actions = len(self._mode_params)
        self._obstacles = []
        
    def _get_initial_state(self) -> np.ndarray:
        """Get initial state with obstacles."""
        # Get base state
        base_state = super()._get_initial_state()
        
        # Generate obstacles
        self._obstacles = []
        for _ in range(self.num_obstacles):
            while True:
                obs = self.np_random.uniform(
                    low=np.array([-self.arena_size/2 + 1, -self.arena_size/2 + 1]),
                    high=np.array([self.arena_size/2 - 1, self.arena_size/2 - 1])
                )
                # Check not too close to start or goal
                dist_start = np.linalg.norm(obs - self._start_pos)
                dist_goal = np.linalg.norm(obs - self._goal_pos)
                if dist_start > 2.0 and dist_goal > 2.0:
                    self._obstacles.append(obs)
                    break
        
        # Flatten obstacles into state
        obs_state = np.array([], dtype=np.float32)
        for obs in self._obstacles:
            obs_state = np.concatenate([obs_state, obs])
        
        return np.concatenate([base_state, obs_state])
    
    def _apply_action(self, state: np.ndarray, action: Tuple[int, np.ndarray]) -> np.ndarray:
        """Apply movement action with obstacle collision."""
        agent_x, agent_y, goal_x, goal_y = state[:4]
        mode, params = action
        
        # Get mode parameters
        max_speed, noise_std = self._mode_params[mode]
        
        # Normalize direction
        direction = params[:2]
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            direction = np.array([1.0, 0.0])
        
        # Get magnitude
        magnitude = params[2] * max_speed
        
        # Add noise
        noise = self.np_random.normal(0, noise_std, size=2)
        
        # Proposed new position
        new_x = agent_x + (direction[0] * magnitude + noise[0])
        new_y = agent_y + (direction[1] * magnitude + noise[1])
        
        # Check obstacle collisions (unless jumping)
        if mode != 4:  # Not jumping
            for obs in self._obstacles:
                dist = np.sqrt((new_x - obs[0])**2 + (new_y - obs[1])**2)
                if dist < self.obstacle_radius:
                    # Collision - push back
                    push_dir = np.array([new_x - obs[0], new_y - obs[1]])
                    push_norm = np.linalg.norm(push_dir)
                    if push_norm > 0:
                        push_dir = push_dir / push_norm
                        new_x = obs[0] + push_dir[0] * self.obstacle_radius
                        new_y = obs[1] + push_dir[1] * self.obstacle_radius
        
        # Clip to arena bounds
        new_x = np.clip(new_x, -self.arena_size/2, self.arena_size/2)
        new_y = np.clip(new_y, -self.arena_size/2, self.arena_size/2)
        
        # Reconstruct state with obstacles
        obs_state = state[4:]  # Keep obstacle positions
        return np.array([new_x, new_y, goal_x, goal_y, *obs_state], dtype=np.float32)
    
    def _compute_reward(self, state: np.ndarray, action: Tuple[int, np.ndarray], next_state: np.ndarray) -> float:
        """Compute reward with obstacle penalty."""
        agent_x, agent_y, goal_x, goal_y = next_state[:4]
        
        dist = np.sqrt((agent_x - goal_x)**2 + (agent_y - goal_y)**2)
        
        # Success reward
        if dist <= self.goal_radius:
            return 100.0
        
        # Obstacle proximity penalty
        obstacle_penalty = 0.0
        for obs in self._obstacles:
            obs_dist = np.sqrt((agent_x - obs[0])**2 + (agent_y - obs[1])**2)
            if obs_dist < self.obstacle_radius * 2:
                obstacle_penalty -= (self.obstacle_radius * 2 - obs_dist)
        
        return -dist + obstacle_penalty - 0.01
    
    def _check_failure(self, state: np.ndarray) -> bool:
        """Check if stuck in obstacle."""
        agent_x, agent_y = state[:2]
        for obs in self._obstacles:
            dist = np.sqrt((agent_x - obs[0])**2 + (agent_y - obs[1])**2)
            if dist < self.obstacle_radius * 0.5:
                return True
        return False
