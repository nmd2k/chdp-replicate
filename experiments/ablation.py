#!/usr/bin/env python3
"""
Ablation study script for CHDP.
Evaluates variants:
- w/o Diffusion Policy (deterministic)
- w/o Codebook (argmax selection)
- w/o Sequential Update (concurrent)
- w/o Both
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


class AblationCHDPTrainer:
    """Trainer for CHDP ablation variants."""
    
    def __init__(self, config: dict, env_name: str, ablation_variant: dict,
                 seed: int, log_dir: str):
        self.config = config
        self.env_name = env_name
        self.ablation_variant = ablation_variant
        self.seed = seed
        self.log_dir = log_dir
        
        set_seed(seed)
        
        self.env = create_environment(env_name, seed=seed)
        self.eval_env = create_environment(env_name, seed=seed + 1000)
        
        state_dim = self.env.state_dim
        action_dim = self.env.action_dim
        param_dim = self.env.param_dim
        
        self.agent = self._create_agent(state_dim, action_dim, param_dim, ablation_variant)
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
        
    def _create_agent(self, state_dim: int, action_dim: int, param_dim: int, 
                      variant: dict):
        """Create CHDP agent with ablation variant."""
        from agents.chdp import CHDPAgent
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return CHDPAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            param_dim=param_dim,
            latent_dim=self.config["architecture"]["latent_dim"],
            hidden_dim=self.config["architecture"]["hidden_dim"],
            num_layers=self.config["architecture"]["num_layers"],
            codebook_size=self.config["architecture"]["codebook_size"] if variant["codebook"] else 1,
            codebook_dim=self.config["architecture"]["codebook_dim"],
            diffusion_steps=self.config["diffusion"]["steps"],
            eta=self.config["dql"]["eta"],
            num_samples=self.config["dql"]["num_samples"],
            gamma=self.config["training"]["gamma"],
            tau=self.config["training"]["tau"],
            lr_actor=self.config["training"]["lr_actor"],
            lr_critic=self.config["training"]["lr_critic"],
            lr_codebook=self.config["training"]["lr_codebook"],
            sequential_update=variant["sequential_update"],
            use_diffusion=variant["diffusion_policy"],
            device=device
        )
    
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
        
        variant_name = self.ablation_variant["name"]
        pbar = tqdm(total=self.total_steps, desc=f"Training {variant_name}")
        
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
            "variant": variant_name,
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


def run_ablation(variant: dict, env_name: str, seed: int,
                 config: dict, results_dir: str) -> dict:
    """Run single ablation experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(results_dir, f"{variant['name']}_{env_name}_seed{seed}_{timestamp}")
    
    trainer = AblationCHDPTrainer(config, env_name, variant, seed, log_dir)
    results = trainer.train()
    trainer.save_results(os.path.join(log_dir, "results.csv"))
    trainer.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run CHDP ablation studies")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml",
                        help="Path to config file")
    parser.add_argument("--variant", type=str, default=None,
                        help="Specific ablation variant to run (default: all)")
    parser.add_argument("--env", type=str, default=None,
                        help="Specific environment to run (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Specific seed to run (default: all)")
    parser.add_argument("--results-dir", type=str, default="results/ablation",
                        help="Directory to save results")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    variants = config["ablation"]
    environments = config["environments"]
    seeds = config["seeds"]
    
    if args.variant:
        variants = [v for v in variants if v["name"] == args.variant]
    
    if args.env:
        environments = [e for e in environments if e["name"] == args.env]
    
    if args.seed is not None:
        seeds = [args.seed]
    
    all_results = {}
    
    for variant in variants:
        variant_name = variant["name"]
        print(f"\n{'='*60}")
        print(f"Ablation Variant: {variant_name}")
        print(f"  Diffusion Policy: {variant['diffusion_policy']}")
        print(f"  Codebook: {variant['codebook']}")
        print(f"  Sequential Update: {variant['sequential_update']}")
        print(f"{'='*60}")
        
        variant_results = {}
        
        for env_config in environments:
            env_name = env_config["name"]
            print(f"\n  Environment: {env_name}")
            
            env_scores = []
            
            for seed in seeds:
                print(f"    Running seed {seed}...")
                result = run_ablation(variant, env_name, seed, config, args.results_dir)
                env_scores.append(result["final_success_rate"])
            
            mean_score = np.mean(env_scores)
            std_score = np.std(env_scores)
            
            variant_results[env_name] = {
                "mean": mean_score,
                "std": std_score,
                "seeds": env_scores
            }
            
            print(f"    {env_name}: {mean_score:.1f} ± {std_score:.1f}")
        
        all_results[variant_name] = variant_results
    
    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Summary saved to:", summary_path)
    print(f"{'='*60}")
    
    print("\nAblation Study Summary (Table 2 format):")
    print("-" * 80)
    print(f"{'Variant':<20} {'Goal':<12} {'Platform':<12} {'Catch':<12} {'HM-4':<12}")
    print("-" * 80)
    
    for variant_name, results in all_results.items():
        row = f"{variant_name:<20}"
        for env in ["goal", "platform", "catch_point", "hard_move_4"]:
            if env in results:
                r = results[env]
                row += f" {r['mean']:.1f} ± {r['std']:.1f}  ".ljust(12)
            else:
                row += " " * 12
        print(row)
    
    print("-" * 80)


if __name__ == "__main__":
    main()
