import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def generate_valid_states(num_states: int):
    """
    Generates an array of valid observation states for the environment.

    Args:
        num_states (int): The number of states to generate.

    Returns:
        list: A list of numpy arrays, where each array is a valid observation.
    """
    print(f"Generating {num_states} valid random states for evaluation...")
    
    min_pos = config.GROUND_HEIGHT + config.BALL_RADIUS
    max_pos = config.SCREEN_HEIGHT - config.BALL_RADIUS
    
    # Generate random values for each component of the observation space
    heights = np.random.uniform(min_pos, max_pos, num_states)
    velocities = np.random.uniform(-50, 50, num_states)  # A reasonable range for velocity
    target_heights = np.random.uniform(min_pos, max_pos, num_states)
    
    # Calculate the fourth component (distance to target)
    distances = target_heights - heights
    
    # Assemble the observations into the correct format (list of numpy arrays)
    observations = [
        np.array([h, v, th, d], dtype=np.float32) 
        for h, v, th, d in zip(heights, velocities, target_heights, distances)
    ]
    
    print("State generation complete.")
    return observations

if __name__ == '__main__':
    TEST_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    NUM_CALCULATIONS = 1000
    
    # 1. Generate a consistent set of states for all agents to be tested on
    states_to_test = generate_valid_states(NUM_CALCULATIONS)
    
    # 2. Find all trained agent models in the directory
    agent_files = [f for f in os.listdir(TEST_DIRECTORY) if f.endswith('.zip')]
    
    if not agent_files:
        print(f"No trained agents (.zip files) found in '{TEST_DIRECTORY}'.")
        sys.exit()
        
    results = {}
    print("\n--- Starting Computational Time Evaluation ---")
    
    # 3. Loop through each agent, load it, and time its predictions
    for agent_file in agent_files:
        model_path = os.path.join(TEST_DIRECTORY, agent_file)
        print(f"Evaluating: {agent_file}")
        
        agent = PPO.load(model_path)
        
        start_time = time.perf_counter()
        for obs in states_to_test:
            _ = agent.predict(obs, deterministic=True)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / NUM_CALCULATIONS) * 1000  # Convert to milliseconds
        results[agent_file] = avg_time_ms
        print(f"  -> Average time: {avg_time_ms:.4f} ms per calculation.")

    # 4. Save the results to a file
    results_path = os.path.join(TEST_DIRECTORY, "compute_time_results.txt")
    with open(results_path, 'w') as f:
        f.write("PPO Agent Computational Time Results\n")
        f.write("="*40 + "\n")
        for agent_name, avg_time in sorted(results.items(), key=lambda item: item[1]):
            f.write(f"{agent_name:<40}: {avg_time:.4f} ms\n")
            
    print(f"\nResults saved to {results_path}")