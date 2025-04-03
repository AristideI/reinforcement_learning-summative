import numpy as np
import pygame
import random
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Constants
GRID_SIZE = 10
CELL_SIZE = 60
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)


class DroneRescueEnv:
    def __init__(self):
        # Initialize environment
        self.grid_size = GRID_SIZE
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.max_battery = 100

        # Environment elements
        self.drone_pos = None
        self.battery_level = None
        self.person_pos = None
        self.safe_zone_pos = None
        self.obstacles = None
        self.carrying_person = None

        # Initialize pygame for visualization
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Drone Search and Rescue")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        # Reset environment
        self.reset()

    def reset(self):
        # Clear grid
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # Place drone at a random position
        self.drone_pos = [
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1),
        ]
        self.grid[self.drone_pos[0], self.drone_pos[1]] = 1

        # Reset battery
        self.battery_level = self.max_battery

        # Place person at a random position different from drone
        while True:
            self.person_pos = [
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            ]
            if self.person_pos != self.drone_pos:
                break
        self.grid[self.person_pos[0], self.person_pos[1]] = 2

        # Place safe zone at a random position different from drone and person
        while True:
            self.safe_zone_pos = [
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            ]
            if (
                self.safe_zone_pos != self.drone_pos
                and self.safe_zone_pos != self.person_pos
            ):
                break
        self.grid[self.safe_zone_pos[0], self.safe_zone_pos[1]] = 3

        # Place obstacles
        self.obstacles = []
        num_obstacles = random.randint(5, 10)
        for _ in range(num_obstacles):
            while True:
                obs_pos = [
                    random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1),
                ]
                if (
                    obs_pos != self.drone_pos
                    and obs_pos != self.person_pos
                    and obs_pos != self.safe_zone_pos
                    and obs_pos not in self.obstacles
                ):
                    self.obstacles.append(obs_pos)
                    self.grid[obs_pos[0], obs_pos[1]] = 4
                    break

        # Reset status
        self.carrying_person = False

        # Return initial state
        return self._get_state()

    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left, 4=pickup, 5=dropoff
        reward = 0
        done = False
        info = {}

        # Reduce battery for any action
        self.battery_level -= 1
        reward -= 0.1  # Penalty for battery consumption

        # Handle movement actions
        if action < 4:  # Movement actions
            # Store previous position
            prev_pos = self.drone_pos.copy()

            # Update position based on action
            if action == 0 and self.drone_pos[0] > 0:  # Move up
                self.drone_pos[0] -= 1
            elif action == 1 and self.drone_pos[1] < self.grid_size - 1:  # Move right
                self.drone_pos[1] += 1
            elif action == 2 and self.drone_pos[0] < self.grid_size - 1:  # Move down
                self.drone_pos[0] += 1
            elif action == 3 and self.drone_pos[1] > 0:  # Move left
                self.drone_pos[1] -= 1

            # Check for out of bounds
            if (
                self.drone_pos[0] < 0
                or self.drone_pos[0] >= self.grid_size
                or self.drone_pos[1] < 0
                or self.drone_pos[1] >= self.grid_size
            ):
                self.drone_pos = prev_pos  # Revert to previous position
                reward -= 5  # Penalty for going out of bounds
                info["out_of_bounds"] = True

            # Check for collision with obstacle
            if self.drone_pos in self.obstacles:
                self.drone_pos = prev_pos  # Revert to previous position
                reward -= 1  # Penalty for hitting obstacle
                info["hit_obstacle"] = True
            else:
                reward += 5  # Reward for avoiding obstacles

            # If carrying a person and moving towards safe zone, give reward
            if self.carrying_person:
                prev_dist = abs(prev_pos[0] - self.safe_zone_pos[0]) + abs(
                    prev_pos[1] - self.safe_zone_pos[1]
                )
                curr_dist = abs(self.drone_pos[0] - self.safe_zone_pos[0]) + abs(
                    self.drone_pos[1] - self.safe_zone_pos[1]
                )
                if curr_dist < prev_dist:
                    reward += 2  # Reward for moving toward safe zone with person

        # Handle pickup action
        elif action == 4:  # Pickup action
            if not self.carrying_person:
                # Check if person is at the same position or adjacent
                if self.drone_pos == self.person_pos or (
                    abs(self.drone_pos[0] - self.person_pos[0]) <= 1
                    and abs(self.drone_pos[1] - self.person_pos[1]) <= 1
                ):
                    self.carrying_person = True
                    reward += 10  # Reward for picking up person
                    info["pickup"] = True
                    # Person is now being carried, remove from grid
                    self.grid[self.person_pos[0], self.person_pos[1]] = 0

        # Handle dropoff action
        elif action == 5:  # Dropoff action
            if self.carrying_person:
                # Check if at safe zone
                if self.drone_pos == self.safe_zone_pos or (
                    abs(self.drone_pos[0] - self.safe_zone_pos[0]) <= 1
                    and abs(self.drone_pos[1] - self.safe_zone_pos[1]) <= 1
                ):
                    self.carrying_person = False
                    reward += 20  # Big reward for successful rescue
                    done = True  # Mission complete
                    info["rescue_complete"] = True

        # Check if battery depleted
        if self.battery_level <= 0:
            done = True
            reward -= 10  # Penalty for running out of battery
            info["battery_depleted"] = True

        # Update grid with new drone position
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.grid[self.drone_pos[0], self.drone_pos[1]] = 1

        # Restore obstacles in grid
        for obs in self.obstacles:
            self.grid[obs[0], obs[1]] = 4

        # Restore person in grid if not carried
        if not self.carrying_person:
            self.grid[self.person_pos[0], self.person_pos[1]] = 2

        # Restore safe zone in grid
        self.grid[self.safe_zone_pos[0], self.safe_zone_pos[1]] = 3

        return self._get_state(), reward, done, info

    def _get_state(self):
        # Returns the current state as a flat array
        state = np.zeros(6)
        state[0] = self.drone_pos[0] / self.grid_size  # Normalized x position
        state[1] = self.drone_pos[1] / self.grid_size  # Normalized y position
        state[2] = self.battery_level / self.max_battery  # Normalized battery
        state[3] = 1 if self.carrying_person else 0  # Carrying status

        if not self.carrying_person:
            # Distance to person (if not carrying)
            state[4] = (
                abs(self.drone_pos[0] - self.person_pos[0])
                + abs(self.drone_pos[1] - self.person_pos[1])
            ) / (2 * self.grid_size)
        else:
            # Distance to safe zone (if carrying)
            state[4] = (
                abs(self.drone_pos[0] - self.safe_zone_pos[0])
                + abs(self.drone_pos[1] - self.safe_zone_pos[1])
            ) / (2 * self.grid_size)

        # Obstacle presence in nearby cells (simplified)
        nearby_obstacle = 0
        for obs in self.obstacles:
            if (
                abs(self.drone_pos[0] - obs[0]) <= 1
                and abs(self.drone_pos[1] - obs[1]) <= 1
            ):
                nearby_obstacle = 1
                break
        state[5] = nearby_obstacle

        return state

    def render(self):
        # Clear screen
        self.screen.fill(WHITE)

        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pygame.draw.rect(
                    self.screen,
                    BLACK,
                    (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    1,
                )

        # Draw obstacles
        for obs in self.obstacles:
            pygame.draw.rect(
                self.screen,
                GRAY,
                (obs[1] * CELL_SIZE, obs[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
            )

        # Draw safe zone
        pygame.draw.rect(
            self.screen,
            GREEN,
            (
                self.safe_zone_pos[1] * CELL_SIZE,
                self.safe_zone_pos[0] * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            ),
        )

        # Draw person (if not carried)
        if not self.carrying_person:
            pygame.draw.circle(
                self.screen,
                BLUE,
                (
                    self.person_pos[1] * CELL_SIZE + CELL_SIZE // 2,
                    self.person_pos[0] * CELL_SIZE + CELL_SIZE // 2,
                ),
                CELL_SIZE // 4,
            )

        # Draw drone
        drone_color = ORANGE if self.carrying_person else RED
        pygame.draw.circle(
            self.screen,
            drone_color,
            (
                self.drone_pos[1] * CELL_SIZE + CELL_SIZE // 2,
                self.drone_pos[0] * CELL_SIZE + CELL_SIZE // 2,
            ),
            CELL_SIZE // 3,
        )

        # Draw battery level
        battery_text = f"Battery: {self.battery_level}%"
        text_surface = self.font.render(battery_text, True, BLACK)
        self.screen.blit(text_surface, (10, 10))

        # Draw status
        status_text = "Carrying Person" if self.carrying_person else "Searching"
        text_surface = self.font.render(status_text, True, BLACK)
        self.screen.blit(text_surface, (10, 40))

        # Update display
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()


# DQN Model
class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQNModel(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.model(next_state)).item()

            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# PPO Model
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1),
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, state, action):
        state = torch.FloatTensor(state)
        probs = self.actor(state)
        dist = Categorical(probs)

        action_logprobs = dist.log_prob(torch.tensor(action))
        dist_entropy = dist.entropy()

        state_value = self.critic(state)

        return action_logprobs, state_value, dist_entropy


# PPO Agent
class PPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4

        self.policy = ActorCritic(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.002)

        self.policy_old = ActorCritic(state_size, action_size)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def act(self, state):
        return self.policy_old.act(state)

    def update(self, memory):
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list to tensor
        old_states = torch.tensor(memory.states).float()
        old_actions = torch.tensor(memory.actions).float()
        old_logprobs = torch.tensor(memory.logprobs).float()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # Final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())


# PPO Memory
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


# Training function for DQN
def train_dqn(env, episodes=1000, batch_size=32, render=True):
    state_size = 6  # Size of our state representation
    action_size = 6  # Number of possible actions
    agent = DQNAgent(state_size, action_size)
    scores = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if render:
                env.render()

            # Decide action
            action = agent.act(state)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Remember experience
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Train (experience replay)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                print(
                    f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}"
                )
                scores.append(total_reward)
                break

        # Decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    return scores


# Training function for PPO
def train_ppo(env, episodes=1000, update_timestep=2000, render=True):
    state_size = 6  # Size of our state representation
    action_size = 6  # Number of possible actions
    agent = PPOAgent(state_size, action_size)
    memory = Memory()
    scores = []

    time_step = 0

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if render:
                env.render()

            time_step += 1

            # Select action
            action, logprob = agent.act(state)
            next_state, reward, done, info = env.step(action)

            # Save in memory
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            state = next_state
            total_reward += reward

            # Update if its time
            if time_step % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                time_step = 0

            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}")
                scores.append(total_reward)
                break

    return scores


# Run simulation with DQN
def run_dqn_simulation():
    env = DroneRescueEnv()
    scores = train_dqn(env, episodes=10, render=True)
    env.close()
    return scores


# Run simulation with PPO
def run_ppo_simulation():
    env = DroneRescueEnv()
    scores = train_ppo(env, episodes=100, render=True)
    env.close()
    return scores


# Run the simulation
if __name__ == "__main__":
    # Choose algorithm: "dqn" or "ppo"
    algorithm = "dqn"

    if algorithm == "dqn":
        scores = run_dqn_simulation()
    else:
        scores = run_ppo_simulation()

    print(f"Average score: {sum(scores)/len(scores)}")
