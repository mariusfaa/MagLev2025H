import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import multiprocessing
from typing import Callable

def create_ppo_agent(env_id, n_envs=None, net_arch=None, activation_fn=nn.ReLU):
    """
    Creates and returns a PPO agent using a vectorized environment for parallel training.
    :param env_id: The class of the environment to instantiate.
    :param n_envs: The number of parallel environments to use. If None, uses the number of CPU cores.
    :param net_arch: A list of integers specifying the number of units in each layer of the actor and critic networks. If None, defaults to [64, 64].
    :param activation_fn: The activation function to use in the networks (e.g., nn.ReLU, nn.Tanh).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device} ---")

    if net_arch is None:
        net_arch = [64, 64]

    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch),
        activation_fn=activation_fn
    )
    
    # Use multiple environments for parallel data collection
    if n_envs is None:
        n_envs = multiprocessing.cpu_count()
    print(f"--- Using {n_envs} parallel environments for training ---")
    vec_env = make_vec_env(env_id, n_envs=n_envs)

    initial_lr = 0.0004
    final_lr = 0.0001
    lr_schedule = linear_schedule(initial_lr, final_lr)

    # Instantiate the PPO agent with the 'MlpPolicy' string and the policy arguments.
    model = PPO("MlpPolicy", vec_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="./ppo_ball_tensorboard/",
                learning_rate=lr_schedule,
                n_steps=2048, 
                batch_size=128,
                ent_coef=0.01)
    
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



def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: The initial learning rate.
    :param final_value: The final learning rate.
    :return: A function that takes the current progress remaining (1.0 -> 0.0)
             and returns the learning rate.
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1.0 to 0.0 over the course of training.
        """
        return final_value + (initial_value - final_value) * progress_remaining

    return func