# CHDP Reproduction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.0.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Reproducing results from "CHDP: Cooperative Hybrid Diffusion Policies for Reinforcement Learning in Parameterized Action Space" (AAAI 2026)**

📄 **Paper:** [arXiv:2601.05675](https://arxiv.org/abs/2601.05675)  
🎯 **Goal:** Reproduce SOTA results on 8 PAMDP benchmarks with up to **19.3% improvement** over prior methods

---

## 🎯 Key Results

| Environment | CHDP Target | Prior SOTA (HyAR) | Improvement |
|-------------|-------------|-------------------|-------------|
| **Hard Goal** | **79.5 ± 5.0** | 60.2 ± 5.0 | **+19.3%** |
| Platform | **99.7 ± 0.2** | 96.6 ± 2.2 | +3.1% |
| Catch Point | **93.8 ± 0.6** | 86.6 ± 0.9 | +7.2% |
| Hard Move (n=6) | **93.9 ± 1.0** | 92.3 ± 0.6 | +1.6% |
| Hard Move (n=10) | **79.8 ± 5.4** | 69.0 ± 5.6 | +10.8% |

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd chdp-reproduction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Test Installation

```bash
# Verify all environments work
python tests/test_environments.py
```

### Train CHDP

```bash
# Train on Hard Goal environment
python experiments/run_chdp.py \
  --env HardGoal \
  --config configs/chdp_default.yaml \
  --seed 0

# Train on all 8 environments (recommended for full reproduction)
python experiments/run_chdp.py \
  --env all \
  --num-seeds 5
```

### Evaluate Baselines

```bash
# Compare with prior SOTA (HyAR-TD3)
python experiments/run_baselines.py \
  --env HardGoal \
  --algo hyar_td3 \
  --seed 0

# Run all baselines
python experiments/run_baselines.py \
  --env all \
  --num-seeds 5
```

---

## 📦 Project Structure

```
chdp-reproduction/
├── src/
│   ├── environments/          # PAMDP benchmark environments
│   │   ├── base_pamdp.py      # Base PAMDP class
│   │   ├── platform.py        # Platform environment
│   │   ├── goal.py            # Goal environment
│   │   ├── hard_goal.py       # Hard Goal environment
│   │   ├── catch_point.py     # Catch Point environment
│   │   └── hard_move.py       # Hard Move (n=4,6,8,10)
│   │
│   ├── agents/                # RL agents
│   │   ├── chdp_agent.py      # Main CHDP agent ⭐
│   │   ├── replay_buffer.py   # Experience replay
│   │   ├── trainer.py         # Training loop
│   │   └── baselines/         # Baseline algorithms
│   │       ├── pdqn_td3.py    # PDQN-TD3
│   │       ├── pa_td3.py      # PA-TD3
│   │       ├── hhqn_td3.py    # HHQN-TD3
│   │       ├── hppo.py        # HPPO
│   │       └── hyar_td3.py    # HyAR-TD3 (prior SOTA)
│   │
│   ├── models/                # Neural network components
│   │   ├── diffusion.py       # Diffusion process (Eq. 1-2)
│   │   ├── codebook.py        # Q-guided codebook (Eq. 12)
│   │   ├── noise_predictor.py # ε_θd, ε_θc networks
│   │   └── q_network.py       # Double Q-learning critics
│   │
│   └── utils/                 # Utilities
│       └── __init__.py        # Seed, logging, checkpoints
│
├── configs/                   # Configuration files
│   ├── chdp_default.yaml      # Default CHDP hyperparameters
│   └── experiment.yaml        # Experiment settings
│
├── experiments/               # Training scripts
│   ├── run_chdp.py            # Main CHDP training
│   ├── run_baselines.py       # Baseline training
│   └── ablation.py            # Ablation studies
│
├── tests/                     # Test scripts
│   └── test_environments.py   # Environment verification
│
├── results/                   # Output directory (created on first run)
│   ├── logs/                  # TensorBoard logs
│   ├── models/                # Saved checkpoints
│   └── csv/                   # CSV results
│
├── README.md                  # This file
├── PROJECT_STATUS.md          # Detailed status & implementation notes
├── requirements.txt           # Python dependencies
└── .gitignore
```

---

## 🎓 Background

### What is Hybrid Action Space?

Hybrid (parameterized) action space combines:
- **Discrete choices** (e.g., jump left/right, select tool A/B/C)
- **Continuous parameters** (e.g., jump force, tool position)

**Example:** In robot soccer, you choose:
- Discrete: Which foot to kick with (left/right)
- Continuous: Kick force (0-100N), angle (-45° to 45°)

### Why CHDP?

**Challenges in hybrid action spaces:**
1. **Limited expressiveness** - Traditional Gaussian/deterministic policies can't capture multi-modal distributions
2. **Poor scalability** - Combinatorial explosion in high-dimensional discrete spaces

**CHDP solutions:**
1. **Diffusion policies** - Expressive enough to model complex multi-modal action distributions
2. **Q-guided codebook** - Embeds high-dimensional discrete space into compact latent space
3. **Sequential update** - Prevents optimization conflicts between discrete and continuous policies

---

## 🔬 Method Details

### CHDP Architecture

```
State s ──→ Discrete Policy π_θd ──→ Latent e ──→ Codebook ──→ Codeword e_k
                                              (VQ)
                                                      │
                                                      ↓
                                              Continuous Policy π_θc
                                              (conditioned on e_k)
                                                      │
                                                      ↓
                                              Continuous action a^c
```

### Key Equations

**1. Diffusion Loss (Eq. 2)**
```
ℒ_d(θ) = 𝔼[||ε - ε_θ(√ᾱᵢa₀ + √(1-ᾱᵢ)ε, s, i)||²]
```

**2. DQL Objective (Eq. 3-4)**
```
ℒ(θ) = ℒ_d(θ) + α × ℒ_q(θ)
ℒ_q(θ) = -𝔼[Q_φ(s, a₀)]
```

**3. Sequential Update (Eq. 7-9)**
```
Step 1: ℒ(θ_d) = ℒ_d(θ_d) - α × 𝔼[Q_φ(s, e, a^c)]
Step 2: ℒ(θ_c, ζ) = ℒ_d(θ_c) + α × ℒ_q(θ_c, ζ)
```

**4. Vector Quantization (Eq. 12)**
```
k = argmin_k ||e - e_k||²
```

### Default Hyperparameters

```yaml
# Diffusion
diffusion_steps: 15
noise_schedule: "cosine"

# Architecture
latent_dim: 8          # Codebook embedding dimension
hidden_dim: 256        # Hidden layer size
num_layers: 3          # Number of MLP layers

# Training
batch_size: 256
buffer_size: 100000
gamma: 0.99            # Discount factor
tau: 0.005             # Soft update coefficient
lr_actor: 0.0001
lr_critic: 0.0003
lr_codebook: 0.0001

# DQL
eta: 5.0               # Q-loss weight

# Evaluation
eval_episodes: 5
eval_freq: 10000       # Evaluate every 10k steps
total_steps: 500000    # 500k environment steps
```

---

## 📊 Environments

### 8 PAMDP Benchmarks

| Environment | Description | Discrete Actions | Continuous Params |
|-------------|-------------|------------------|-------------------|
| **Platform** | Navigate platforms with jumps | 2 (jump left/right) | 1 (force) |
| **Goal** | Reach target position | 3 (movement modes) | 2 (direction, speed) |
| **Hard Goal** | Challenging goal-reaching | 4 (movement types) | 2 (angle, power) |
| **Catch Point** | Intercept moving target | 3 (catch strategies) | 2 (timing, position) |
| **Hard Move (n=4)** | 4 actuators | 2⁴ = 16 | 4 (per actuator) |
| **Hard Move (n=6)** | 6 actuators | 2⁶ = 64 | 6 (per actuator) |
| **Hard Move (n=8)** | 8 actuators | 2⁸ = 256 | 8 (per actuator) |
| **Hard Move (n=10)** | 10 actuators | 2¹⁰ = 1024 | 10 (per actuator) |

### Environment Usage

```python
from src.environments import HardGoalEnv

# Create environment
env = HardGoalEnv()

# Reset
state = env.reset()

# Step with hybrid action
discrete_action = 1  # e.g., kick with right foot
continuous_params = [0.75, 0.5]  # e.g., force=0.75, angle=0.5
next_state, reward, done, info = env.step(discrete_action, continuous_params)

# Access action spaces
print(f"Discrete actions: {env.discrete_action_space.n}")
print(f"Continuous dims: {env.continuous_action_space.shape[0]}")
```

---

## 🧪 Experiments

### Reproduce Main Results (Table 1)

```bash
# Train CHDP on all environments with 5 seeds
python experiments/run_chdp.py \
  --env all \
  --num-seeds 5 \
  --config configs/chdp_default.yaml

# Train all baselines
python experiments/run_baselines.py \
  --env all \
  --num-seeds 5
```

**Expected runtime:** ~48 hours on single GPU (RTX 3090)

### Run Ablation Study (Table 2)

```bash
# Test without diffusion policy (deterministic)
python experiments/ablation.py \
  --env HardGoal \
  --variant w/o_diffusion \
  --seed 0

# Test without codebook
python experiments/ablation.py \
  --env HardMove \
  --variant w/o_codebook \
  --seed 0

# Test without sequential update
python experiments/ablation.py \
  --env HardGoal \
  --variant w/o_sequential \
  --seed 0

# Run all ablations
python experiments/ablation.py \
  --env all \
  --variants all
```

### Visualize Results

```bash
# Generate learning curves (Figure 4)
python experiments/plot_results.py \
  --results-dir results/csv \
  --output-dir results/figures \
  --plot-type learning_curves

# Generate comparison table (Table 1)
python experiments/plot_results.py \
  --results-dir results/csv \
  --output-dir results/tables \
  --plot-type table
```

---

## 📈 Monitoring Training

### TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir results/logs

# Open browser to http://localhost:6006
```

### CSV Results

Results are saved to `results/csv/<env>/<algo>/<seed>/metrics.csv`:

```csv
step,success_rate,avg_reward,q_loss,actor_loss,codebook_loss
10000,0.45,2.3,0.52,1.23,0.08
20000,0.62,3.1,0.41,1.05,0.06
...
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Environment test fails**
```bash
# Verify installation
python -c "import torch; import numpy; import gymnasium"

# Run with verbose output
python tests/test_environments.py --verbose
```

**2. Training diverges**
- Reduce learning rate: `--lr-actor 5e-5 --lr-critic 1e-4`
- Increase batch size: `--batch-size 512`
- Check reward scaling: environments should have rewards in [-1, 1] range

**3. CUDA out of memory**
- Reduce batch size: `--batch-size 128`
- Reduce hidden dimension: Edit `configs/chdp_default.yaml`

**4. Poor performance**
- Increase training steps: `--total-steps 1000000`
- Tune η (Q-loss weight): Try `--eta 3.0` or `--eta 10.0`
- Ensure proper exploration: Add action noise during training

### Getting Help

- Check [PROJECT_STATUS.md](PROJECT_STATUS.md) for implementation details
- Review paper equations vs. code implementation
- Open an issue with error logs and configuration

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{liu2026chdp,
  title={CHDP: Cooperative Hybrid Diffusion Policies for Reinforcement Learning in Parameterized Action Space},
  author={Liu, Bingyi and He, Jinbo and Shi, Haiyong and Wang, Enshu and Han, Weizhen and Hao, Jingxiang and Wang, Peixi and Zhang, Zhuangzhuang},
  journal={arXiv preprint arXiv:2601.05675},
  year={2026}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m 'Add my feature'`
4. Push to branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Reproduction by:** [Your Name/Team]

**Original CHDP paper:**
- Bingyi Liu, Jinbo He, Haiyong Shi, Enshu Wang, Weizhen Han, Jingxiang Hao, Peixi Wang, Zhuangzhuang Zhang
- Accepted to AAAI 2026

---

## 🗓️ Changelog

- **2026-04-22**: Initial reproduction release
  - ✅ All 8 PAMDP environments implemented
  - ✅ CHDP agent with sequential update
  - ✅ 5 baseline algorithms
  - ✅ Training and evaluation scripts
  - ✅ Ablation study framework

---

## 📞 Contact

- **Issues:** [GitHub Issues](https://github.com/your-repo/chdp-reproduction/issues)
- **Email:** your.email@example.com

---

**Happy Reproducing! 🚀**
