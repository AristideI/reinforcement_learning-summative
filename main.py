import os
import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

# Import our modules
from environment.custom_env import DroneRescueEnv
from training.dqn_training import train_dqn
from training.pg_training import train_ppo

# Register the custom environment
gym.register(
    id="DroneRescue-v0",
    entry_point="environment.custom_env:DroneRescueEnv",
    max_episode_steps=200,
)


def visualize_env():
    """
    Visualize the environment without any training
    """
    env = gym.make("DroneRescue-v0", render_mode="human")
    env.reset()

    print("Press Ctrl+C to exit visualization")
    try:
        while True:
            action = np.random.randint(0, 6)  # Random action
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
    except KeyboardInterrupt:
        print("Visualization stopped by user")
    finally:
        env.close()


def run_dqn_training(timesteps=50000, render_eval=True):
    """
    Run DQN training
    """
    print("Starting DQN training...")
    env = gym.make("DroneRescue-v0")

    # Train the model
    model = train_dqn(env, total_timesteps=timesteps, verbose=1)

    # Evaluate the model
    if render_eval:
        eval_env = gym.make("DroneRescue-v0", render_mode="human")
    else:
        eval_env = gym.make("DroneRescue-v0")

    # mean_reward, std_reward = evaluate_dqn(
    #     model, eval_env, n_eval_episodes=5, render=render_eval
    # )
    # print(
    #     f"DQN training completed! Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}"
    # )

    return model


def run_ppo_training(timesteps=50000, render_eval=True):
    """
    Run PPO training
    """
    print("Starting PPO training...")
    env = gym.make("DroneRescue-v0")

    # Train the model
    model = train_ppo(env, total_timesteps=timesteps, verbose=1)

    # Evaluate the model
    if render_eval:
        eval_env = gym.make("DroneRescue-v0", render_mode="human")
    else:
        eval_env = gym.make("DroneRescue-v0")

    # mean_reward, std_reward = evaluate_ppo(
    #     model, eval_env, n_eval_episodes=5, render=render_eval
    # )
    # print(
    #     f"PPO training completed! Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}"
    # )

    return model


def compare_models(
    dqn_model_path="models/dqn/best_model.zip",
    ppo_model_path="models/pg/best_model.zip",
    n_eval_episodes=10,
):
    """
    Compare the performance of DQN and PPO models
    """
    # Create evaluation environment
    eval_env = Monitor(gym.make("DroneRescue-v0"))

    # Load models
    dqn_model = DQN.load(dqn_model_path)
    ppo_model = PPO.load(ppo_model_path)

    # Evaluate models
    print("Evaluating DQN model...")
    dqn_rewards, _ = evaluate_policy(
        dqn_model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        return_episode_rewards=True,
    )

    print("Evaluating PPO model...")
    ppo_rewards, _ = evaluate_policy(
        ppo_model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        return_episode_rewards=True,
    )

    # Calculate statistics
    dqn_mean = np.mean(dqn_rewards)
    dqn_std = np.std(dqn_rewards)
    ppo_mean = np.mean(ppo_rewards)
    ppo_std = np.std(ppo_rewards)

    print(f"DQN: Mean reward = {dqn_mean:.2f} +/- {dqn_std:.2f}")
    print(f"PPO: Mean reward = {ppo_mean:.2f} +/- {ppo_std:.2f}")

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.boxplot([dqn_rewards, ppo_rewards], labels=["DQN", "PPO"])
    plt.title("DQN vs PPO Performance Comparison")
    plt.ylabel("Episode Reward")
    plt.savefig("model_comparison.png")
    plt.close()

    # Create a bar chart with error bars
    plt.figure(figsize=(8, 6))
    algorithms = ["DQN", "PPO"]
    means = [dqn_mean, ppo_mean]
    stds = [dqn_std, ppo_std]

    plt.bar(algorithms, means, yerr=stds, capsize=10, color=["blue", "orange"])
    plt.title("Performance Comparison: DQN vs PPO")
    plt.ylabel("Mean Episode Reward")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("model_comparison_bar.png")
    plt.close()

    return dqn_rewards, ppo_rewards


def main():
    """
    Main function to parse arguments and run the appropriate training or evaluation
    """
    parser = argparse.ArgumentParser(
        description="Drone Rescue RL Training and Evaluation"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="visualize",
        choices=[
            "visualize",
            "train_dqn",
            "train_ppo",
            "train_both",
            "evaluate",
            "compare",
        ],
        help="Mode to run: visualize, train_dqn, train_ppo, train_both, evaluate, or compare",
    )
    parser.add_argument(
        "--timesteps", type=int, default=50000, help="Number of timesteps for training"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Disable rendering during evaluation"
    )

    args = parser.parse_args()

    # Create directories for models
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("models/pg", exist_ok=True)

    # Run the selected mode
    if args.mode == "visualize":
        visualize_env()

    elif args.mode == "train_dqn":
        run_dqn_training(timesteps=args.timesteps, render_eval=not args.no_render)

    elif args.mode == "train_ppo":
        run_ppo_training(timesteps=args.timesteps, render_eval=not args.no_render)

    elif args.mode == "train_both":
        print("Training both DQN and PPO models...")
        run_dqn_training(timesteps=args.timesteps, render_eval=not args.no_render)
        run_ppo_training(timesteps=args.timesteps, render_eval=not args.no_render)

    elif args.mode == "evaluate":
        # Check if models exist
        dqn_path = "models/dqn/best_model.zip"
        ppo_path = "models/pg/best_model.zip"

        if os.path.exists(dqn_path):
            print("Evaluating DQN model...")
            eval_env = gym.make(
                "DroneRescue-v0", render_mode=None if args.no_render else "human"
            )
            dqn_model = DQN.load(dqn_path)
            mean_reward, _ = evaluate_policy(dqn_model, eval_env, n_eval_episodes=5)
            print(f"DQN mean reward: {mean_reward:.2f}")
        else:
            print(f"DQN model not found at {dqn_path}")

        if os.path.exists(ppo_path):
            print("Evaluating PPO model...")
            eval_env = gym.make(
                "DroneRescue-v0", render_mode=None if args.no_render else "human"
            )
            ppo_model = PPO.load(ppo_path)
            mean_reward, _ = evaluate_policy(ppo_model, eval_env, n_eval_episodes=5)
            print(f"PPO mean reward: {mean_reward:.2f}")
        else:
            print(f"PPO model not found at {ppo_path}")

    elif args.mode == "compare":
        # Check if models exist
        dqn_path = "models/dqn/best_model.zip"
        ppo_path = "models/pg/best_model.zip"

        if os.path.exists(dqn_path) and os.path.exists(ppo_path):
            compare_models(dqn_path, ppo_path, n_eval_episodes=10)
        else:
            print("Models not found. Please train both models first.")
            if not os.path.exists(dqn_path):
                print(f"DQN model not found at {dqn_path}")
            if not os.path.exists(ppo_path):
                print(f"PPO model not found at {ppo_path}")


if __name__ == "__main__":
    main()
