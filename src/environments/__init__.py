"""PAMDP environments from the CHDP paper (arXiv:2601.05675)."""

from .base_pamdp import BasePAMDP
from .platform import PlatformEnv
from .goal import GoalEnv, HardGoalEnv
from .catch_point import CatchPointEnv
from .hard_move import HardMoveEnv

__all__ = [
    "BasePAMDP",
    "PlatformEnv",
    "GoalEnv",
    "HardGoalEnv",
    "CatchPointEnv",
    "HardMoveEnv",
]

# Environment registry for gymnasium
def register_environments():
    """Register all PAMDP environments with gymnasium."""
    import gymnasium as gym
    
    gym.register(
        id="PAMDP-Platform-v0",
        entry_point="src.environments:PlatformEnv",
        max_episode_steps=100,
    )
    
    gym.register(
        id="PAMDP-Goal-v0",
        entry_point="src.environments:GoalEnv",
        max_episode_steps=100,
    )
    
    gym.register(
        id="PAMDP-HardGoal-v0",
        entry_point="src.environments:HardGoalEnv",
        max_episode_steps=100,
    )
    
    gym.register(
        id="PAMDP-CatchPoint-v0",
        entry_point="src.environments:CatchPointEnv",
        max_episode_steps=100,
    )
    
    # Register HardMove variants
    for n in [4, 6, 8, 10]:
        gym.register(
            id=f"PAMDP-HardMove-{n}-v0",
            entry_point="src.environments:HardMoveEnv",
            kwargs={"n": n},
            max_episode_steps=100,
        )
