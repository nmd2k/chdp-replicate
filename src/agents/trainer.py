"""Training loop for CHDP agent."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

from .chdp_agent import CHDPAgent
from .replay_buffer import ReplayBuffer


class Trainer:
    """
    Trainer for CHDP agent with sequential update scheme.
    
    Implements the training loop from the paper:
    1. Collect experience in environment
    2. Store transitions in replay buffer
    3. Perform sequential updates (discrete -> continuous+codebook -> critic)
    """
    
    def __init__(
        self,
        agent: CHDPAgent,
        replay_buffer: ReplayBuffer,
        batch_size: int = 256,
        update_freq: int = 1,
        grad_steps: int = 1,
        min_buffer_size: int = 1000,
        device: str = "cpu"
    ):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.grad_steps = grad_steps
        self.min_buffer_size = min_buffer_size
        self.device = torch.device(device)
        
        # Training metrics
        self.loss_history: Dict[str, List[float]] = {
            "loss_discrete": [],
            "loss_continuous": [],
            "vq_loss": [],
            "critic_loss": []
        }
        self.reward_history: List[float] = []
    
    def train(
        self,
        env,
        total_steps: int,
        eval_freq: int = 10000,
        eval_episodes: int = 5,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Args:
            env: Gym environment
            total_steps: Total training steps
            eval_freq: Evaluation frequency
            eval_episodes: Number of evaluation episodes
            save_path: Path to save checkpoints
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        state = env.reset()
        latent = None
        episode_reward = 0
        
        start_time = time.time()
        
        for step in range(total_steps):
            # =========================================================================
            # Action Selection
            # =========================================================================
            action, latent = self.agent.select_action(
                state, latent, deterministic=False
            )
            
            # =========================================================================
            # Environment Step
            # =========================================================================
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # =========================================================================
            # Store Transition in Replay Buffer
            # =========================================================================
            self.replay_buffer.add(
                state=state,
                latent=latent,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            # Reset if done
            if done:
                self.reward_history.append(episode_reward)
                episode_reward = 0
                state = env.reset()
                latent = None
            else:
                state = next_state
            
            # =========================================================================
            # Training Step
            # =========================================================================
            if step % self.update_freq == 0 and self.replay_buffer.is_ready(self.min_buffer_size):
                for _ in range(self.grad_steps):
                    batch = self.replay_buffer.sample(self.batch_size)
                    losses = self.agent.update(batch)
                    
                    # Record losses
                    for key, value in losses.items():
                        self.loss_history[key].append(value)
            
            # =========================================================================
            # Evaluation
            # =========================================================================
            if step % eval_freq == 0 and step > 0:
                eval_reward = self._evaluate(env, eval_episodes)
                
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"Step {step}/{total_steps} | "
                          f"Eval Reward: {eval_reward:.2f} | "
                          f"Time: {elapsed:.1f}s")
                
                if save_path:
                    self.agent.save(f"{save_path}_step_{step}.pt")
        
        # Final evaluation
        final_reward = self._evaluate(env, eval_episodes)
        if verbose:
            print(f"Training complete! Final eval reward: {final_reward:.2f}")
        
        if save_path:
            self.agent.save(f"{save_path}_final.pt")
        
        return {
            "losses": self.loss_history,
            "rewards": self.reward_history
        }
    
    def _evaluate(
        self,
        env,
        num_episodes: int = 5
    ) -> float:
        """
        Evaluate agent performance.
        
        Args:
            env: Gym environment
            num_episodes: Number of episodes to average
            
        Returns:
            Average episode reward
        """
        rewards = []
        
        for _ in range(num_episodes):
            state = env.reset()
            latent = None
            episode_reward = 0
            done = False
            
            while not done:
                action, latent = self.agent.select_action(
                    state, latent, deterministic=True
                )
                state, reward, done, _ = env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        return np.mean(rewards)
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics."""
        stats = {}
        
        for key, values in self.loss_history.items():
            if values:
                stats[f"{key}_mean"] = np.mean(values[-100:])
                stats[f"{key}_std"] = np.std(values[-100:])
        
        if self.reward_history:
            stats["reward_mean"] = np.mean(self.reward_history[-100:])
            stats["reward_max"] = max(self.reward_history)
        
        stats["buffer_size"] = len(self.replay_buffer)
        
        return stats


def create_agent_and_buffer(
    state_dim: int,
    action_dim: int,
    latent_dim: int = 32,
    num_codes: int = 64,
    hidden_dim: int = 256,
    buffer_capacity: int = 100000,
    device: str = "cpu"
) -> Tuple[CHDPAgent, ReplayBuffer]:
    """
    Factory function to create agent and replay buffer.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        latent_dim: Dimension of latent space
        num_codes: Number of codebook entries
        hidden_dim: Hidden layer dimension
        buffer_capacity: Replay buffer capacity
        device: Device to run on
        
    Returns:
        Tuple of (agent, replay_buffer)
    """
    agent = CHDPAgent(
        state_dim=state_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
        num_codes=num_codes,
        hidden_dim=hidden_dim,
        device=device
    )
    
    replay_buffer = ReplayBuffer(
        capacity=buffer_capacity,
        state_dim=state_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
        device=device
    )
    
    return agent, replay_buffer
