import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from environment.custom_env import DroneRescueEnv


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            ep_rew_mean = self.model.logger.name_to_value["rollout/ep_rew_mean"]

            # Here, we're assuming that ep_rew_mean is a single float value
            if ep_rew_mean is not None:
                mean_reward = ep_rew_mean  # Use ep_rew_mean directly as the reward

                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}"
                    )

                # New best model, save the agent
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


def train_dqn(
    env_id,
    total_timesteps=100000,
    learning_rate=0.0001,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    max_grad_norm=10,
    verbose=1,
):
    """
    Train a DQN model on the DroneRescue environment

    Args:
        env_id: Environment ID or environment instance
        total_timesteps: Number of training timesteps
        learning_rate: Learning rate
        buffer_size: Size of the replay buffer
        learning_starts: Number of steps before learning starts
        batch_size: Batch size for training
        tau: Soft update coefficient
        gamma: Discount factor
        train_freq: Update the model every train_freq steps
        gradient_steps: How many gradient steps to do after each rollout
        target_update_interval: Update the target network every target_update_interval steps
        exploration_fraction: Fraction of total timesteps for exploration
        exploration_initial_eps: Initial value of random action probability
        exploration_final_eps: Final value of random action probability
        max_grad_norm: Maximum norm for gradient clipping
        verbose: Verbosity level

    Returns:
        Trained model
    """
    # Create log dir
    log_dir = "models/dqn/"
    os.makedirs(log_dir, exist_ok=True)

    # Create callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    # Create and train the model
    model = DQN(
        "MlpPolicy",
        env_id,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        max_grad_norm=max_grad_norm,
        tensorboard_log=log_dir,
        verbose=verbose,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the final model
    model.save(os.path.join(log_dir, "final_model"))

    return model


def evaluate_dqn(model, env, n_eval_episodes=10, render=False):
    # Check if the environment supports the 'render' method (no need for 'render_mode' setting)
    if render and hasattr(env, "render"):
        # Use the default render method for rendering
        env.render()

    # Evaluate the model using the evaluate_policy function
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return mean_reward, std_reward


if __name__ == "__main__":
    # Create environment
    env = gym.make("DroneRescue-v0")

    # Train DQN model
    model = train_dqn(env, total_timesteps=50000, verbose=1)

    # # Evaluate model
    # evaluate_dqn(model, env, n_eval_episodes=10, render=True)
