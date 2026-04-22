#!/usr/bin/env python3
"""
Baseline training script for CHDP reproduction.
Implements SAC, TD3, DDPG, PPO, Diffusion-BC, Diffusion-RL, and CQL baselines.
"""

import os
import sys
import argparse
import yaml
import json
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from environments.base_pamdp import create_environment
from agents.replay_buffer import ReplayBuffer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaselineTrainer:
    """Trainer for baseline agents."""
    
    def __init__(self, config: dict, env_name: str, baseline_name: str, 
                 seed: int, log_dir: str):
        self.config = config
        self.env_name = env_name
        self.baseline_name = baseline_name
        self.seed = seed
        self.log_dir = log_dir
        
        set_seed(seed)
        
        self.env = create_environment(env_name, seed=seed)
        self.eval_env = create_environment(env_name, seed=seed + 1000)
        
        state_dim = self.env.state_dim
        action_dim = self.env.action_dim
        param_dim = self.env.param_dim
        
        self.agent = self._create_agent(baseline_name, state_dim, action_dim, param_dim)
        self.replay_buffer = ReplayBuffer(
            capacity=config["training"]["buffer_size"],
            state_dim=state_dim,
            action_dim=action_dim,
            param_dim=param_dim
        )
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.results = []
        
        self.total_steps = config["training"]["total_steps"]
        self.batch_size = config["training"]["batch_size"]
        self.eval_freq = config["evaluation"]["frequency"]
        self.eval_episodes = config["evaluation"]["episodes"]
        self.warmup_steps = config["training"]["warmup_steps"]
        
    def _create_agent(self, name: str, state_dim: int, action_dim: int, param_dim: int):
        """Create baseline agent."""
        from agents.baselines import (
            SACAgent, TD3Agent, DDPGAgent, PPOAgent,
            DiffusionBCAgent, DiffusionRLAgent, CQLAgent
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hidden_dim = self.config["architecture"]["hidden_dim"]
        gamma = self.config["training"]["gamma"]
        tau = self.config["training"]["tau"]
        
        agents = {
            "sac": lambda: SACAgent(
                state_dim, action_dim, param_dim, hidden_dim,
                gamma=gamma, tau=tau,
                lr=self.config["training"]["lr_actor"],
                device=device
            ),
            "td3": lambda: TD3Agent(
                state_dim, action_dim, param_dim, hidden_dim,
                gamma=gamma, tau=tau,
                lr_actor=self.config["training"]["lr_actor"],
                lr_critic=self.config["training"]["lr_critic"],
                device=device
            ),
            "ddpg": lambda: DDPGAgent(
                state_dim, action_dim, param_dim, hidden_dim,
                gamma=gamma, tau=tau,
                lr_actor=self.config["training"]["lr_actor"],
                lr_critic=self.config["training"]["lr_critic"],
                device=device
            ),
            "ppo": lambda: PPOAgent(
                state_dim, action_dim, param_dim, hidden_dim,
                gamma=gamma,
                lr=self.config["training"]["lr_actor"],
                device=device
            ),
            "diffusion_bc": lambda: DiffusionBCAgent(
                state_dim, action_dim, param_dim, hidden_dim,
                diffusion_steps=self.config["diffusion"]["steps"],
                lr=self.config["training"]["lr_actor"],
                device=device
            ),
            "diffusion_rl": lambda: DiffusionRLAgent(
                state_dim, action_dim, param_dim, hidden_dim,
                diffusion_steps=self.config["diffusion"]["steps"],
                eta=self.config["dql"]["eta"],
                gamma=gamma,
                lr=self.config["training"]["lr_actor"],
                device=device
            ),
            "cql": lambda: CQLAgent(
                state_dim, action_dim, param_dim, hidden_dim,
                gamma=gamma, tau=tau,
                cql_alpha=1.0,
                lr=self.config["training"]["lr_actor"],
                device=device
            )
        }
        
        if name not in agents:
            raise ValueError(f"Unknown baseline: {name}")
        
        return agents[name]()
    
    def evaluate(self) -> float:
        """Evaluate agent on environment."""
        success_count = 0
        
        for _ in range(self.eval_episodes):
            state, _ = self.eval_env.reset()
            done = False
            
            while not done:
                action, param = self.agent.select_action(state, deterministic=True)
                state, _, terminated, truncated, info = self.eval_env.step(action, param)
                done = terminated or truncated
            
            if info.get("success", False):
                success_count += 1
        
        return success_count / self.eval_episodes
    
    def train(self) -> dict:
        """Run training loop."""
        state, _ = self.env.reset()
        episode_reward = 0
        step = 0
        
        pbar = tqdm(total=self.total_steps, desc=f"Training {self.baseline_name}")
        
        while step < self.total_steps:
            if step < self.warmup_steps:
                action, param = self.env.action_space.sample()
            else:
                action, param = self.agent.select_action(state, deterministic=False)
            
            next_state, reward, terminated, truncated, info = self.env.step(action, param)
            done = terminated or truncated
            
            self.replay_buffer.add(state, action, param, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            step += 1
            
            if step >= self.warmup_steps:
                self.agent.update(self.replay_buffer, self.batch_size)
            
            if done:
                state, _ = self.env.reset()
                episode_reward = 0
            
            pbar.update(1)
            
            if step % self.eval_freq == 0:
                success_rate = self.evaluate()
                
                self.results.append({
                    "step": step,
                    "success_rate": success_rate,
                    "episode_reward": episode_reward
                })
                
                self.writer.add_scalar(
                    "evaluation/success_rate", 
                    success_rate, 
                    step
                )
                
                tqdm.write(f"Step {step}: Success Rate = {success_rate:.3f}")
        
        pbar.close()
        
        final_scores = [r["success_rate"] for r in self.results[-5:]]
        final_score = np.mean(final_scores)
        
        return {
            "env_name": self.env_name,
            "baseline": self.baseline_name,
            "seed": self.seed,
            "final_success_rate": final_score,
            "all_evaluations": self.results
        }
    
    def save_results(self, filepath: str):
        """Save results to CSV."""
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "success_rate", "episode_reward"])
            writer.writeheader()
            writer.writerows(self.results)
    
    def close(self):
        """Clean up resources."""
        self.writer.close()
        self.env.close()
        self.eval_env.close()


def run_baseline(baseline_name: str, env_name: str, seed: int, 
                 config: dict, results_dir: str) -> dict:
    """Run single baseline experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(results_dir, f"{baseline_name}_{env_name}_seed{seed}_{timestamp}")
    
    trainer = BaselineTrainer(config, env_name, baseline_name, seed, log_dir)
    results = trainer.train()
    trainer.save_results(os.path.join(log_dir, "results.csv"))
    trainer.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train baseline agents")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml",
                        help="Path to config file")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Specific baseline to run (default: all)")
    parser.add_argument("--env", type=str, default=None,
                        help="Specific environment to run (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Specific seed to run (default: all)")
    parser.add_argument("--results-dir", type=str, default="results/baselines",
                        help="Directory to save results")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    baselines = config["baselines"]
    environments = config["environments"]
    seeds = config["seeds"]
    
    if args.baseline:
        baselines = [b for b in baselines if b == args.baseline]
    
    if args.env:
        environments = [e for e in environments if e["name"] == args.env]
    
    if args.seed is not None:
        seeds = [args.seed]
    
    all_results = {}
    
    for baseline_name in baselines:
        print(f"\n{'='*60}")
        print(f"Baseline: {baseline_name}")
        print(f"{'='*60}")
        
        baseline_results = {}
        
        for env_config in environments:
            env_name = env_config["name"]
            print(f"\n  Environment: {env_name}")
            
            env_scores = []
            
            for seed in seeds:
                print(f"    Running seed {seed}...")
                result = run_baseline(baseline_name, env_name, seed, config, args.results_dir)
                env_scores.append(result["final_success_rate"])
            
            mean_score = np.mean(env_scores)
            std_score = np.std(env_scores)
            
            baseline_results[env_name] = {
                "mean": mean_score,
                "std": std_score,
                "seeds": env_scores
            }
            
            print(f"    {env_name}: {mean_score:.1f} ± {std_score:.1f}")
        
        all_results[baseline_name] = baseline_results
    
    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Summary saved to:", summary_path)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
