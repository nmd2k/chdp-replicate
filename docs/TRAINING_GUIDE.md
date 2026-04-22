# CHDP Training Guide

**Complete guide for training CHDP and baseline algorithms**

---

## 🎯 Quick Start

### Single Environment Training

```bash
# Train CHDP on Hard Goal
python experiments/run_chdp.py \
  --env HardGoal \
  --seed 0 \
  --config configs/chdp_default.yaml
```

### Full Reproduction

```bash
# Train CHDP on all 8 environments with 5 seeds each
python experiments/run_chdp.py \
  --env all \
  --num-seeds 5 \
  --config configs/chdp_default.yaml

# Expected runtime: ~48 hours on RTX 3090
```

---

## 📋 Configuration Options

### Command-Line Arguments

```bash
python experiments/run_chdp.py \
  --env HardGoal \                    # Environment name or 'all'
  --seed 0 \                          # Random seed
  --config configs/chdp_default.yaml \ # Config file
  --total-steps 500000 \              # Total environment steps
  --eval-freq 10000 \                 # Evaluate every N steps
  --eval-episodes 5 \                 # Episodes per evaluation
  --save-freq 50000 \                 # Save checkpoint every N steps
  --device cuda \                     # Device (cuda/cpu/mps)
  --log-dir results/logs \            # TensorBoard logs
  --model-dir results/models \        # Checkpoint directory
  --csv-dir results/csv               # CSV results directory
```

### Hyperparameter Overrides

```bash
# Override specific hyperparameters
python experiments/run_chdp.py \
  --env HardGoal \
  --eta 10.0 \                        # Q-loss weight
  --diffusion-steps 20 \              # Diffusion sampling steps
  --latent-dim 16 \                   # Codebook embedding dim
  --hidden-dim 512 \                  # Hidden layer size
  --lr-actor 5e-5 \                   # Actor learning rate
  --lr-critic 3e-4 \                  # Critic learning rate
  --batch-size 512                    # Batch size
```

---

## 🔧 Environment-Specific Recommendations

### Easy Environments (Platform, Goal)

```bash
# These converge quickly - can reduce training steps
python experiments/run_chdp.py \
  --env Platform \
  --total-steps 300000 \
  --eval-freq 5000
```

### Medium Environments (Hard Goal, Catch Point)

```bash
# Standard configuration works well
python experiments/run_chdp.py \
  --env HardGoal \
  --total-steps 500000 \
  --eta 5.0
```

### Hard Environments (Hard Move series)

```bash
# High-dimensional discrete space - need more training
python experiments/run_chdp.py \
  --env HardMove \
  --n-actuators 10 \
  --total-steps 1000000 \
  --latent-dim 16 \                   # Larger codebook
  --eta 10.0                          # More Q-guidance
```

---

## 📊 Monitoring Training

### Real-Time Monitoring with TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir results/logs --port 6006

# Open browser to http://localhost:6006
```

**Key metrics to watch:**
- `success_rate` - Should increase over time
- `q_loss` - Should decrease and stabilize
- `actor_loss` - May fluctuate, trend downward
- `codebook_loss` - Should decrease
- `avg_reward` - Should correlate with success rate

### CSV Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/csv/HardGoal/chdp/seed_0/metrics.csv')

# Plot success rate
plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['success_rate'])
plt.xlabel('Steps')
plt.ylabel('Success Rate')
plt.title('CHDP Training on Hard Goal')
plt.grid(True)
plt.savefig('learning_curve.png')
```

---

## 🧪 Debugging Training Issues

### Training Diverges

**Symptoms:** Success rate drops to 0, Q-loss explodes

**Solutions:**
```bash
# 1. Reduce learning rates
python experiments/run_chdp.py \
  --lr-actor 1e-5 \
  --lr-critic 1e-4 \
  --lr-codebook 1e-5

# 2. Reduce Q-loss weight
python experiments/run_chdp.py \
  --eta 1.0

# 3. Increase batch size
python experiments/run_chdp.py \
  --batch-size 512
```

### Slow Convergence

**Symptoms:** Success rate increases very slowly

**Solutions:**
```bash
# 1. Increase Q-loss weight
python experiments/run_chdp.py \
  --eta 10.0

# 2. More frequent updates
python experiments/run_chdp.py \
  --train-freq 1 \       # Update every step
  --grad-steps 2         # 2 gradient steps per update

# 3. More exploration
# Edit chdp_agent.py: Add action noise during early training
```

### Out of Memory

**Symptoms:** CUDA out of memory error

**Solutions:**
```bash
# 1. Reduce batch size
python experiments/run_chdp.py \
  --batch-size 128

# 2. Reduce model size
python experiments/run_chdp.py \
  --hidden-dim 128 \
  --latent-dim 4

# 3. Use CPU (slower but works)
python experiments/run_chdp.py \
  --device cpu
```

---

## 🎓 Hyperparameter Tuning

### Grid Search Example

```bash
#!/bin/bash
# scripts/grid_search.sh

etas=(3.0 5.0 10.0)
diffusion_steps=(10 15 20)
latent_dims=(8 16)

for eta in "${etas[@]}"; do
  for steps in "${diffusion_steps[@]}"; do
    for dim in "${latent_dims[@]}"; do
      python experiments/run_chdp.py \
        --env HardGoal \
        --seed 0 \
        --eta $eta \
        --diffusion-steps $steps \
        --latent-dim $dim \
        --log-dir results/logs/tuning/eta${eta}_steps${steps}_dim${dim}
    done
  done
done
```

### Recommended Search Spaces

| Hyperparameter | Range | Default |
|----------------|-------|---------|
| η (eta) | 1.0 - 20.0 | 5.0 |
| Diffusion steps | 10 - 50 | 15 |
| Latent dim | 4 - 32 | 8 |
| Hidden dim | 128 - 512 | 256 |
| Actor LR | 1e-5 - 5e-4 | 1e-4 |
| Critic LR | 1e-4 - 1e-3 | 3e-4 |
| Batch size | 64 - 512 | 256 |

---

## 💾 Checkpointing & Resuming

### Automatic Checkpointing

Checkpoints are saved automatically every `--save-freq` steps:

```
results/models/
├── HardGoal/
│   ├── chdp_seed_0_step_50000.pt
│   ├── chdp_seed_0_step_100000.pt
│   └── ...
```

### Resume Training

```bash
python experiments/run_chdp.py \
  --env HardGoal \
  --resume results/models/HardGoal/chdp_seed_0_step_100000.pt \
  --total-steps 500000  # Continue to 500k
```

### Load Checkpoint for Evaluation

```bash
python experiments/evaluate.py \
  --env HardGoal \
  --checkpoint results/models/HardGoal/chdp_seed_0_step_500000.pt \
  --episodes 100
```

---

## 📈 Evaluation Protocol

### Standard Evaluation

```bash
# Evaluate trained model
python experiments/evaluate.py \
  --env HardGoal \
  --checkpoint results/models/HardGoal/chdp_seed_0_final.pt \
  --episodes 100 \
  --deterministic  # Use deterministic actions
```

### Compute Final Metrics

```python
# experiments/compute_metrics.py
import numpy as np
import glob

# Load all seed results
success_rates = []
for seed in range(5):
    df = pd.read_csv(f'results/csv/HardGoal/chdp/seed_{seed}/metrics.csv')
    # Average final 5 evaluations
    final_sr = df['success_rate'].tail(5).mean()
    success_rates.append(final_sr)

# Report mean ± std
mean = np.mean(success_rates)
std = np.std(success_rates)
print(f"Success Rate: {mean:.1f} ± {std:.1f}")
```

---

## 🔄 Multi-Seed Training

### Run 5 Seeds in Parallel

```bash
#!/bin/bash
# scripts/run_multiseed.sh

for seed in 0 1 2 3 4; do
  python experiments/run_chdp.py \
    --env HardGoal \
    --seed $seed \
    --log-dir results/logs/HardGoal/seed_$seed \
    --csv-dir results/csv/HardGoal/seed_$seed &
done

wait
echo "All seeds completed"
```

### Aggregate Results

```bash
python experiments/aggregate_results.py \
  --env HardGoal \
  --algo chdp \
  --seeds 5
```

---

## 🎯 Baseline Training

### Train All Baselines

```bash
# HyAR-TD3 (prior SOTA)
python experiments/run_baselines.py \
  --env HardGoal \
  --algo hyar_td3 \
  --seed 0

# PDQN-TD3
python experiments/run_baselines.py \
  --env HardGoal \
  --algo pdqn_td3 \
  --seed 0

# All baselines, all seeds
python experiments/run_baselines.py \
  --env all \
  --num-seeds 5
```

### Baseline Hyperparameters

```yaml
# configs/baselines.yaml
pdqn_td3:
  hidden_dim: 256
  lr: 0.0003
  policy_noise: 0.2
  policy_freq: 2

pa_td3:
  hidden_dim: 256
  lr: 0.0003
  
hhqn_td3:
  hidden_dim: 256
  lr: 0.0003

hppo:
  hidden_dim: 256
  lr: 0.0003
  epochs: 10
  batch_size: 64

hyar_td3:
  latent_dim: 64
  hidden_dim: 256
  lr: 0.0003
```

---

## 🧪 Ablation Study

### Run All Ablations

```bash
python experiments/ablation.py \
  --env HardGoal \
  --variants w/o_diffusion w/o_codebook w/o_sequential w/o_both \
  --num-seeds 5
```

### Individual Ablations

```bash
# Without diffusion policy (deterministic)
python experiments/ablation.py \
  --env HardGoal \
  --variant w/o_diffusion \
  --seed 0

# Without codebook (argmax selection)
python experiments/ablation.py \
  --env HardMove \
  --variant w/o_codebook \
  --seed 0

# Without sequential update (concurrent)
python experiments/ablation.py \
  --env HardGoal \
  --variant w/o_sequential \
  --seed 0
```

---

## 📊 Result Visualization

### Generate Learning Curves

```bash
python experiments/plot_results.py \
  --results-dir results/csv \
  --output-dir results/figures \
  --plot-type learning_curves \
  --envs HardGoal Platform CatchPoint
```

### Generate Comparison Table

```bash
python experiments/plot_results.py \
  --results-dir results/csv \
  --output-dir results/tables \
  --plot-type table \
  --format latex  # or 'markdown'
```

### Generate Bar Chart

```bash
python experiments/plot_results.py \
  --results-dir results/csv \
  --output-dir results/figures \
  --plot-type bar_chart \
  --metric success_rate
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** Success rate stays at 0

**Diagnosis:**
```bash
# Check if agent is exploring
python experiments/debug_exploration.py \
  --env HardGoal \
  --checkpoint results/models/.../step_50000.pt

# Check Q-values
python experiments/debug_qvalues.py \
  --env HardGoal \
  --checkpoint results/models/.../step_50000.pt
```

**Issue:** Training is very slow

**Solutions:**
- Reduce `--total-steps` for debugging
- Use GPU: `--device cuda`
- Reduce `--eval-freq` to evaluate less often
- Reduce `--eval-episodes`

**Issue:** Results don't match paper

**Checklist:**
- [ ] Verify environment dynamics match paper
- [ ] Check hyperparameters match config
- [ ] Ensure enough training steps
- [ ] Run with multiple seeds (≥5)
- [ ] Average final 5 evaluations (as per paper)

---

## 📚 Best Practices

1. **Always use multiple seeds** (minimum 3, recommended 5)
2. **Log everything** (TensorBoard + CSV)
3. **Save checkpoints frequently** (every 50k steps)
4. **Monitor training in real-time** (TensorBoard)
5. **Start with small experiments** (100k steps) before full runs
6. **Compare with baselines** on same seeds for fair comparison
7. **Average final 5 evaluations** (as per paper protocol)
8. **Document all hyperparameters** for reproducibility

---

**Happy Training! 🚀**
