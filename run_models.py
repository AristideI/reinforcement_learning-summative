import gym
import numpy as np
from stable_baselines3 import DQN


def load_and_run_model(model_path, env_name):
    # Load environment
    env = gym.make(env_name)

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
    env_name = "./environment/custom_env.py"  # Replace with the actual environment name
    load_and_run_model(model_path, env_name)
