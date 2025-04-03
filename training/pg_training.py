import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
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


def train_ppo(
    env_id,
    total_timesteps=100000,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    normalize_advantage=True,
    verbose=1,
):
    """
    Train a PPO model on the DroneRescue environment

    Args:
        env_id: Environment ID or environment instance
        total_timesteps: Number of training timesteps
        learning_rate: Learning rate
        n_steps: Number of steps to run for each environment per update
        batch_size: Batch size for training
        n_epochs: Number of epochs to train on each update
        gamma: Discount factor
        gae_lambda: Factor for trade-off of bias vs variance in GAE
        clip_range: Clipping parameter for PPO
        normalize_advantage: Whether to normalize advantage
        verbose: Verbosity level

    Returns:
        Trained model
    """
    # Create log dir
    log_dir = "models/pg/"
    os.makedirs(log_dir, exist_ok=True)

    # Create callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    # Create and train the model
    model = PPO(
        "MlpPolicy",
        env_id,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        normalize_advantage=normalize_advantage,
        tensorboard_log=log_dir,
        verbose=verbose,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the final model
    model.save(os.path.join(log_dir, "final_model"))

    return model


def evaluate_ppo(model, env, n_eval_episodes=10, render=False):
    """
    Evaluate a trained PPO model

    Args:
        model: Trained model
        env: Environment to evaluate on
        n_eval_episodes: Number of episodes to evaluate
        render: Whether to render the environment during evaluation

    Returns:
        Mean reward and standard deviation
    """
    # Set render mode if required
    if render and hasattr(env, "render_mode"):
        env.render_mode = "human"

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return mean_reward, std_reward


if __name__ == "__main__":
    # Create environment
    env = gym.make("DroneRescue-v0")

    # Train PPO model
    model = train_ppo(env, total_timesteps=50000, verbose=1)

    # Evaluate model
    evaluate_ppo(model, env, n_eval_episodes=10, render=True)
