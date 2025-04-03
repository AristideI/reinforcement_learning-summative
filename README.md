# Drone Search and Rescue - Reinforcement Learning

This project implements a simulated drone search and rescue mission using reinforcement learning algorithms. Two RL methods are compared: a value-based method (Deep Q-Network, DQN) and a policy-based method (Proximal Policy Optimization, PPO).

## Video Demo



https://github.com/user-attachments/assets/5b2a1704-5cc3-4202-9731-bbb040300ab4



## Environment

The custom environment simulates a drone searching for a person in a grid-based world and rescuing them by taking them to a safe zone. The drone must manage its battery while avoiding obstacles.

### State Space

- Drone position (x, y)
- Battery level
- Carrying status (0 or 1)
- Distance to person or safe zone (depending on carrying status)
- Nearby obstacle presence

### Action Space

- Move Up (0)
- Move Right (1)
- Move Down (2)
- Move Left (3)
- Pickup Person (4)
- Dropoff Person (5)

### Rewards

- -0.1 for each step (battery consumption)
- -5 for trying to go out of bounds
- -1 for hitting an obstacle
- +0.1 for avoiding obstacles
- +2 for moving toward safe zone when carrying person
- +10 for picking up person
- +20 for successful rescue (mission complete)
- -10 for battery depletion

## Environment Visualization

The environment is visualized using PyOpenGL and Pygame. Red circle represents the drone, blue circle represents the person to be rescued, green square represents the safe zone, and gray squares represent obstacles.

![Environment Visualization](environment_visualization.gif)

## Setup and Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/student_name_rl_summative.git
   cd student_name_rl_summative
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Visualize the Environment

```bash
python main.py --mode visualize
```

### Train a DQN Agent

```bash
python main.py --mode train_dqn --timesteps 50000
```

### Train a PPO Agent

```bash
python main.py --mode train_ppo --timesteps 50000
```

### Train Both Agents

```bash
python main.py --mode train_both --timesteps 50000
```

### Evaluate Trained Agents

```bash
python main.py --mode evaluate
```

### Compare Model Performance

```bash
python main.py --mode compare
```

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment implementation
│   ├── rendering.py             # Visualization components using PyOpenGL
├── training/
│   ├── dqn_training.py          # Training script for DQN using SB3
│   ├── pg_training.py           # Training script for PPO/other PG using SB3
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved policy gradient models
├── main.py                      # Entry point for running experiments
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Results

After training both models with the same parameters and comparing their performance, we found that [insert your findings here based on the model comparison plots].

![Model Comparison](model_comparison.png)

## Hyperparameters

### DQN Hyperparameters

- Learning rate: 0.0001
- Buffer size: 10000
- Learning starts: 1000
- Batch size: 64
- Gamma (discount factor): 0.99
- Target update interval: 1000
- Exploration fraction: 0.1
- Initial exploration: 1.0
- Final exploration: 0.05

### PPO Hyperparameters

- Learning rate: 0.0003
- n_steps: 2048
- Batch size: 64
- n_epochs: 10
- Gamma (discount factor): 0.99
- GAE lambda: 0.95
- Clip range: 0.2
- Normalize advantage: True
