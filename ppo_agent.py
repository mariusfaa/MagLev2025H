import torch
import torch.nn as nn
from stable_baselines3 import PPO

def create_ppo_agent(env):
    """
    Creates and returns a PPO agent using a standard policy.
    """
    # Define the network architecture and activation function that were in the custom policy.
    # These are now passed as keyword arguments.
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Even larger network
        activation_fn=nn.ReLU
    )

    # Instantiate the PPO agent with the 'MlpPolicy' string and the policy arguments.
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="./ppo_ball_tensorboard/",
                learning_rate=0.0001,
                n_steps=4096, 
                batch_size=128)
    
    return model

def train_agent(model, timesteps):
    """Trains the agent for a given number of timesteps."""
    print("--- Training PPO Agent ---")
    # The learn method handles the entire training loop
    model.learn(total_timesteps=timesteps)
    print("--- Training Complete ---")
    # Optionally, save the trained model
    model.save("ppo_ball_controller")
    return model

def load_agent(env):
    """Loads a pre-trained agent."""
    print("--- Loading Pre-trained PPO Agent ---")
    model = PPO.load("ppo_ball_controller", env=env)
    return model
