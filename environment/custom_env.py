import gym
import numpy as np
from stable_baselines3 import DQN
import importlib.util
import sys
import os


def load_custom_env(env_path, env_class_name):
    """
    Dynamically load the custom environment from a given file path.
    """
    env_dir = os.path.dirname(env_path)
    env_filename = os.path.basename(env_path).replace(".py", "")

    # Add the directory to sys.path
    sys.path.insert(0, env_dir)

    # Load the module
    spec = importlib.util.spec_from_file_location(env_filename, env_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the environment class
    env_class = getattr(module, env_class_name)

    return env_class()


def load_and_run_model(model_path, env_path, env_class_name):
    # Load custom environment
    env = load_custom_env(env_path, env_class_name)

    # Print environment observation space for debugging
    expected_obs_shape = env.observation_space.shape
    print(f"Expected observation shape from environment: {expected_obs_shape}")

    # Load trained model
    model = DQN.load(model_path)

    # Check model policy expected input shape
    dummy_obs = env.reset()[0]  # Adjust for Gym v26+ returning (obs, info)
    print(f"Actual observation shape received: {dummy_obs.shape}")

    if dummy_obs.shape != expected_obs_shape:
        raise ValueError(
            f"Observation shape mismatch! Model expects {expected_obs_shape}, but got {dummy_obs.shape}"
        )

    # Run model in environment
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)  # Adjust for Gym v26+
        env.render()

    env.close()


if __name__ == "__main__":
    model_path = "models/dqn/final_model.zip"  # Replace with the correct model path
    env_path = "environments/custom_env.py"  # Path to the custom environment file
    env_class_name = "DroneRescueEnv"  # Class name of the custom environment

    load_and_run_model(model_path, env_path, env_class_name)
