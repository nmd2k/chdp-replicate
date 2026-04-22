"""Utilities for CHDP."""

import random
import numpy as np
import torch
import os
from typing import Dict, Any


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    """Soft update target network parameters."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target: torch.nn.Module, source: torch.nn.Module):
    """Hard update target network parameters."""
    target.load_state_dict(source.state_dict())


def save_checkpoint(
    agent: Any,
    optimizer_states: Dict[str, Any],
    step: int,
    path: str,
):
    """Save training checkpoint."""
    checkpoint = {
        "agent_state_dict": agent.state_dict(),
        "optimizer_states": optimizer_states,
        "step": step,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str, device: torch.device):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


class RunningMeanStd:
    """Running mean and standard deviation for normalization."""

    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        """Update running mean and std."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count


class Logger:
    """Simple logger for training metrics."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.data = []
        os.makedirs(log_dir, exist_ok=True)

    def log(self, step: int, metrics: Dict[str, float]):
        """Log metrics at given step."""
        entry = {"step": step, **metrics}
        self.data.append(entry)

    def save(self, filename: str = "metrics.csv"):
        """Save logs to CSV."""
        import csv

        path = os.path.join(self.log_dir, filename)
        if not self.data:
            return

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)

    def get_latest(self, key: str) -> float:
        """Get latest value for a metric."""
        if not self.data:
            return 0.0
        return self.data[-1].get(key, 0.0)
