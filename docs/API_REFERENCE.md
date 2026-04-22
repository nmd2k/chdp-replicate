# API Documentation

**Complete API reference for CHDP reproduction**

---

## Environments (`src/environments`)

### Base Class

#### `base_pamdp.PAMDPEnv`

Base class for all Parameterized Action MDP environments.

```python
from src.environments import PAMDPEnv

class CustomEnv(PAMDPEnv):
    def __init__(self, config=None):
        super().__init__(
            state_dim=...,
            discrete_action_dim=...,
            continuous_param_dim=...,
        )
    
    def reset(self, seed=None):
        """Reset environment and return initial state."""
        ...
        return state
    
    def step(self, discrete_action, continuous_params):
        """Execute hybrid action and return next state, reward, done, info."""
        ...
        return next_state, reward, done, info
```

**Methods:**
- `reset(seed: Optional[int] = None) -> np.ndarray` - Reset environment
- `step(discrete_action: int, continuous_params: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]` - Take action
- `render(mode='human')` - Render environment (optional)
- `close()` - Clean up resources

**Properties:**
- `observation_space: gym.spaces.Box` - State space
- `discrete_action_space: gym.spaces.Discrete` - Discrete action space
- `continuous_action_space: gym.spaces.Box` - Continuous action space

---

### Environment Classes

#### `platform.PlatformEnv`

Platform navigation with jumps.

```python
from src.environments import PlatformEnv

env = PlatformEnv()
state = env.reset()
discrete_action = 0  # Jump left
continuous_params = [0.5]  # Force
next_state, reward, done, info = env.step(discrete_action, continuous_params)
```

**Configuration:**
- State dim: 8
- Discrete actions: 2 (jump left, jump right)
- Continuous params: 1 (force)

---

#### `goal.GoalEnv`

Goal-reaching with movement modes.

```python
from src.environments import GoalEnv

env = GoalEnv()
```

**Configuration:**
- State dim: 6
- Discrete actions: 3 (walk, run, jump)
- Continuous params: 2 (direction, speed)

---

#### `hard_goal.HardGoalEnv`

Challenging goal-reaching.

```python
from src.environments import HardGoalEnv

env = HardGoalEnv()
```

**Configuration:**
- State dim: 8
- Discrete actions: 4 (movement types)
- Continuous params: 2 (angle, power)

---

#### `catch_point.CatchPointEnv`

Intercept moving target.

```python
from src.environments import CatchPointEnv

env = CatchPointEnv()
```

**Configuration:**
- State dim: 10
- Discrete actions: 3 (catch strategies)
- Continuous params: 2 (timing, position)

---

#### `hard_move.HardMoveEnv`

Combinatorial actuator control.

```python
from src.environments import HardMoveEnv

# 4 actuators: 2^4 = 16 discrete actions
env = HardMoveEnv(n_actuators=4)

# 10 actuators: 2^10 = 1024 discrete actions
env = HardMoveEnv(n_actuators=10)
```

**Configuration:**
- State dim: 4n (n = number of actuators)
- Discrete actions: 2^n (on/off for each actuator)
- Continuous params: n (force per actuator)

---

## Models (`src/models`)

### Diffusion Process

#### `diffusion.DiffusionProcess`

```python
from src.models import DiffusionProcess

# Initialize
diffusion = DiffusionProcess(
    n_steps=15,           # Number of diffusion steps
    noise_schedule='cosine',  # 'linear' or 'cosine'
)

# Forward process (add noise)
noisy_action, noise = diffusion.forward_process(
    action=a0,            # Original action
    state=s,              # Condition state
    t=timestep            # Current timestep
)

# Reverse process (denoise)
action_sample = diffusion.reverse_process(
    model=noise_predictor,
    state=s,
    condition=codeword    # Optional conditioning
)

# Compute diffusion loss
loss = diffusion.compute_loss(
    model=noise_predictor,
    action=a0,
    state=s,
    condition=codeword
)
```

**Methods:**
- `forward_process(action, state, t) -> Tuple[noisy_action, noise]` - Add noise
- `reverse_process(model, state, condition) -> action_sample` - Sample from policy
- `compute_loss(model, action, state, condition) -> loss` - Training objective

---

### Q-Guided Codebook

#### `codebook.QGuidedCodebook`

```python
from src.models import QGuidedCodebook

# Initialize
codebook = QGuidedCodebook(
    num_embeddings=64,    # K: number of discrete actions
    embedding_dim=8,      # d_e: latent dimension
    ema_decay=0.99,       # EMA decay for codebook updates
)

# Vector quantization
quantized, indices, commit_loss = codebook.quantize(
    latents=e,            # Continuous latents from discrete policy
    q_values=Q_values     # Optional: Q-values for guidance
)

# Get specific codeword
codeword = codebook.get_embedding(index=k)

# Update codebook with Q-guidance
codebook.update_with_q_guidance(
    q_function=q_net,
    states=states,
    latents=latents
)
```

**Methods:**
- `quantize(latents, q_values=None) -> Tuple[quantized, indices, commit_loss]` - VQ
- `get_embedding(index) -> codeword` - Retrieve codeword
- `update_with_q_guidance(q_function, states, latents)` - Q-guided update

---

### Noise Predictors

#### `noise_predictor.DiscreteNoisePredictor`

```python
from src.models import DiscreteNoisePredictor

# Initialize
predictor = DiscreteNoisePredictor(
    state_dim=8,
    latent_dim=8,
    hidden_dim=256,
    num_layers=3,
)

# Predict noise
noise_pred = predictor(
    noisy_latent=e_t,
    state=s,
    timestep=t
)
```

**Architecture:**
```
Input: [noisy_latent, state, time_emb]
  ↓
MLP (hidden_dim × num_layers)
  ↓
Output: predicted_noise
```

---

#### `noise_predictor.ContinuousNoisePredictor`

```python
from src.models import ContinuousNoisePredictor

# Initialize
predictor = ContinuousNoisePredictor(
    state_dim=8,
    action_dim=2,
    codeword_dim=8,
    hidden_dim=256,
)

# Predict noise (conditioned on codeword)
noise_pred = predictor(
    noisy_action=a^c_t,
    state=s,
    codeword=e_k,
    timestep=t
)
```

**Architecture:**
```
Input: [noisy_action, state, codeword, time_emb]
  ↓
MLP (conditioned on codeword)
  ↓
Output: predicted_noise
```

---

### Q-Networks

#### `q_network.DoubleQNetwork`

```python
from src.models import DoubleQNetwork

# Initialize
q_net = DoubleQNetwork(
    state_dim=8,
    latent_dim=8,
    action_dim=2,
    hidden_dim=256,
)

# Forward pass (get Q-values from both critics)
q1, q2 = q_net(
    state=s,
    latent=e,
    action=a^c
)

# Get minimum (for TD3 target)
q_min = torch.min(q1, q2)

# Update target networks
q_net.soft_update_target(tau=0.005)
```

**Methods:**
- `forward(state, latent, action) -> Tuple[q1, q2]` - Get Q-values
- `soft_update_target(tau)` - Soft update target networks
- `hard_update_target()` - Hard update target networks

---

## Agents (`src/agents`)

### CHDP Agent

#### `chdp_agent.CHDPAgent`

```python
from src.agents import CHDPAgent

# Initialize
agent = CHDPAgent(
    state_dim=8,
    discrete_action_dim=64,
    continuous_action_dim=2,
    latent_dim=8,
    hidden_dim=256,
    diffusion_steps=15,
    eta=5.0,
    gamma=0.99,
    tau=0.005,
    lr_actor=1e-4,
    lr_critic=3e-4,
    lr_codebook=1e-4,
)

# Select action (inference)
discrete_action, continuous_params = agent.select_action(
    state=s,
    explore=True  # Add exploration noise
)

# Train on batch
metrics = agent.update(
    batch={
        'state': states,
        'latent': latents,
        'continuous_params': actions,
        'reward': rewards,
        'next_state': next_states,
        'done': dones,
    }
)

# Save/load
agent.save('checkpoint.pt')
agent.load('checkpoint.pt', device='cuda')
```

**Methods:**
- `select_action(state, explore=True) -> Tuple[discrete_action, continuous_params]` - Action selection
- `update(batch) -> Dict[str, float]` - Training update
- `save(path)` - Save checkpoint
- `load(path, device)` - Load checkpoint
- `state_dict() -> Dict` - Get state dict
- `load_state_dict(state_dict)` - Load state dict

**Sequential Update (internal):**
1. Update discrete policy π_θd (Eq. 7)
2. Update continuous policy π_θc + codebook E_ζ (Eq. 8-9)
3. Update critics Q_φ (Eq. 10-11)

---

### Replay Buffer

#### `replay_buffer.ReplayBuffer`

```python
from src.agents import ReplayBuffer

# Initialize
buffer = ReplayBuffer(
    capacity=100000,
    state_dim=8,
    latent_dim=8,
    action_dim=2,
)

# Add transition
buffer.add(
    state=s,
    latent=e,
    continuous_params=a^c,
    reward=r,
    next_state=s',
    done=d,
)

# Sample batch
batch = buffer.sample(batch_size=256)
# Returns: Dict with keys: 'state', 'latent', 'continuous_params', 'reward', 'next_state', 'done'

# Check if enough samples
if len(buffer) > 1000:
    # Start training
    ...
```

**Methods:**
- `add(state, latent, continuous_params, reward, next_state, done)` - Add transition
- `sample(batch_size) -> Dict` - Sample random batch
- `__len__() -> int` - Get current size

---

### Trainer

#### `trainer.Trainer`

```python
from src.agents import Trainer

# Initialize
trainer = Trainer(
    agent=agent,
    env=env,
    eval_env=eval_env,
    total_steps=500000,
    batch_size=256,
    train_freq=1,
    grad_steps=1,
    eval_freq=10000,
    eval_episodes=5,
    save_freq=50000,
    log_dir='results/logs',
    model_dir='results/models',
    csv_dir='results/csv',
)

# Train
trainer.train(seed=0)
```

**Methods:**
- `train(seed)` - Run training loop
- `evaluate(episodes, deterministic)` - Evaluate agent
- `save_checkpoint(step)` - Save checkpoint
- `log_metrics(step, metrics)` - Log to TensorBoard

---

## Baselines (`src/agents/baselines`)

### PDQN-TD3

```python
from src.agents.baselines import PDQNTD3

agent = PDQNTD3(
    state_dim=8,
    discrete_action_dim=64,
    continuous_param_dim=2,
    hidden_dim=256,
)

action = agent.select_action(state)
losses = agent.train(batch)
```

---

### PA-TD3

```python
from src.agents.baselines import PATD3

agent = PATD3(
    state_dim=8,
    discrete_action_dim=64,
    continuous_param_dim=2,
)
```

---

### HHQN-TD3

```python
from src.agents.baselines import HHQNTD3

agent = HHQNTD3(
    state_dim=8,
    discrete_action_dim=64,
    continuous_param_dim=2,
)
```

---

### HPPO

```python
from src.agents.baselines import HPPO

agent = HPPO(
    state_dim=8,
    discrete_action_dim=64,
    continuous_param_dim=2,
    epochs=10,
    batch_size=64,
)
```

---

### HyAR-TD3

```python
from src.agents.baselines import HyARTD3

agent = HyARTD3(
    state_dim=8,
    discrete_action_dim=64,
    continuous_param_dim=2,
    latent_dim=64,
)
```

---

## Utilities (`src/utils`)

### Seed Management

```python
from src.utils import set_seed

# Set all random seeds
set_seed(42)
```

---

### Device Management

```python
from src.utils import get_device

# Get best available device
device = get_device()  # Returns: cuda, mps, or cpu
```

---

### Network Updates

```python
from src.utils import soft_update, hard_update

# Soft update target network
soft_update(target_net, source_net, tau=0.005)

# Hard update target network
hard_update(target_net, source_net)
```

---

### Checkpointing

```python
from src.utils import save_checkpoint, load_checkpoint

# Save
save_checkpoint(
    agent=agent,
    optimizer_states=opt_states,
    step=100000,
    path='checkpoint.pt',
)

# Load
checkpoint = load_checkpoint('checkpoint.pt', device='cuda')
agent.load_state_dict(checkpoint['agent_state_dict'])
```

---

### Logging

```python
from src.utils import Logger

# Initialize
logger = Logger(log_dir='results/logs')

# Log metrics
logger.log(step=10000, metrics={
    'success_rate': 0.75,
    'q_loss': 0.52,
    'actor_loss': 1.23,
})

# Save to CSV
logger.save('metrics.csv')
```

---

### Normalization

```python
from src.utils import RunningMeanStd

# Initialize
rms = RunningMeanStd(shape=(8,))

# Update with new data
rms.update(new_observations)

# Normalize
normalized = (observations - rms.mean) / np.sqrt(rms.var)
```

---

**End of API Documentation**
