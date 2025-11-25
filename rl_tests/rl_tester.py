import numpy as np
import matplotlib.pyplot as plt
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
import torch.nn as nn
import os
import sys
from typing import Callable

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Assuming BallEnv, config, and plot_data are available from the main project
from environment import BallEnv
import config
from ppo_agent import create_ppo_agent

def train_agent(model, timesteps, save_path):
    """Trains the agent and saves the model."""
    print(f"--- Training PPO Agent for {timesteps} timesteps ---")
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"--- Training Complete. Model saved to {save_path}.zip ---")

def evaluate_agent(model_path, render=True):
    """Evaluates a trained agent and saves the performance data."""
    print(f"--- Evaluating agent from {model_path}.zip ---")
    
    # Create a single environment for evaluation
    eval_env = BallEnv(render_mode='human' if render else None)
    
    # Load the trained agent
    model = PPO.load(model_path, env=eval_env)

    obs, _ = eval_env.reset()
    running = True
    
    # Data logging
    states = []
    actions = []
    references = []

    while running:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)

        # Log data: state (pos, vel), action, reference
        states.append(obs[:2])  # [ball.y, ball.velocity]
        actions.append(action[0])
        references.append(obs[2]) # target_height

        if terminated or truncated:
            print("Episode finished.")
            # In a testing script, we might want to run multiple episodes,
            # but for a single run visualization, one is enough.
            running = False

    eval_env.close()

    # Save the collected data
    data_save_path = f"{model_path}_eval_data.npz"
    np.savez(data_save_path, 
             states=np.array(states), 
             actions=np.array(actions), 
             references=np.array(references))
    print(f"Evaluation data saved to {data_save_path}")

    # Plotting the results
    plot_evaluation_data(data_save_path)

def plot_evaluation_data(file_path):
    """Loads and plots evaluation data."""
    data = np.load(file_path)
    states = data['states']
    actions = data['actions']
    references = data['references']
    
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"PPO Agent Performance ({os.path.basename(file_path)})")
    
    # Plot Height vs. Target
    plt.subplot(2, 1, 1)
    plt.plot(states[:, 0], label='Ball Height')
    plt.plot(references, 'r--', label='Target Height')
    plt.ylabel('Height (pixels)')
    plt.legend()
    plt.grid(True)

    # Plot Control Action
    plt.subplot(2, 1, 2)
    plt.plot(actions, label='Control Input (Force)', color='orange')
    plt.xlabel('Timestep')
    plt.ylabel('Force')
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    # --- Configuration for the test run ---
    NET_ARCH = [128, 128, 128]
    ACTIVATION_FN = nn.Tanh
    TIMESTEPS = config.PPO_TIMESTEPS
    MODEL_NAME = f"ppo_test_arch{NET_ARCH}_act{ACTIVATION_FN.__name__}"
    MODEL_SAVE_PATH = os.path.join("rl_tests", MODEL_NAME)

    # 1. Create Agent
    agent = create_ppo_agent(
        env_id=BallEnv, 
        net_arch=NET_ARCH, 
        activation_fn=ACTIVATION_FN
    )

    # 2. Train Agent
    train_agent(agent, timesteps=TIMESTEPS, save_path=MODEL_SAVE_PATH)

    # 3. Evaluate Agent
    evaluate_agent(MODEL_SAVE_PATH, render=True)