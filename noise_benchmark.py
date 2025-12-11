import numpy as np
import random
import csv
import os
import time
import math
from noise import pnoise1

# Import your existing modules
import config
from ball_simulation import Ball
from p_controller import PController
from mpc_controller import MPCController
from mpc_controller_stoch import MPCControllerStochastic
from mpc_controller_tube import MPCControllerTube
# from mpc_controller_acados import MPCControllerACADOS
# from mpc_controller_stoch_acados import MPCControllerStochasticAcados
# from mpc_controller_tube_acados import MPCControllerTubeAcados

# Try to import PPO, but continue if not available/trained
try:
    from ppo_agent import load_agent
    from environment import BallEnv
    PPO_AVAILABLE = True
except ImportError:
    PPO_AVAILABLE = False
    print("Warning: PPO modules not found. Skipping PPO tests.")

# --- TEST CONFIGURATION ---
TEST_DURATION = 30.0  # Seconds per run
STEPS = int(TEST_DURATION / config.TIME_STEP)
OCTAVES_TO_TEST = [2, 3, 4]
RUNS_PER_SETTING = 10
OUTPUT_DIR = "noise_benchmark_results"


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_noise_step(time_cursor, octaves, base):
    """Replicates the noise generation logic to log the exact value."""
    # Note: pnoise1 returns value roughly in [-1, 1]
    return pnoise1(time_cursor, octaves=octaves, base=base) * config.PERLIN_AMPLITUDE

def run_simulation(controller, controller_name, octave, run_id):
    """
    Runs a single simulation episode.
    Returns: A list of data rows and the root mean squared error (RMSE).
    """
    
    # 1. Setup Environment
    ball = Ball(config.SCREEN_WIDTH / 2, config.STARTING_HEIGHT)
    
    # Random base for this specific run
    base_seed = random.randint(0, 100)
    
    # Noise tracking
    noise_time_cursor = 0.0
    
    # Data storage
    data_rows = []
    total_sq_error = 0.0
    
    # # If PPO, we need the gymnasium environment wrapper
    # ppo_env = None
    # ppo_obs = None
    # if controller_name == "PPO":
    #     ppo_env = BallEnv(render_mode=None)
    #     ppo_obs, _ = ppo_env.reset()
    #     # Hack to sync the env ball with our local ball tracking if needed,
    #     # but usually PPO uses the env's internal ball. 
    #     # To ensure fair comparison with exact noise injection, we should ideally
    #     # use the same loop. However, PPO expects `env.step()`.
    #     # For simplicity in this script, we will treat PPO slightly differently 
    #     # or assume the standard loop if the agent allows direct prediction.
    #     # Stable Baselines3 expects env interaction. 
    #     # Let's stick to the manual loop for P/MPC and skip PPO in this specific custom loop
    #     # unless we wrap the manual loop as an env. 
    #     pass

    target = config.TARGET_HEIGHT
    
    for step in range(STEPS):
        current_time = step * config.TIME_STEP
        
        # 1. Get Control Action
        # if controller_name == "P_Controller":
        #     # PController expects: (y, target, vel)
        #     force = controller.get_action(ball.y, target, ball.velocity)
            
        if controller_name == "MPC" or controller_name == "MPC_Stochastic" or controller_name == "MPC_Tube":
            # MPC expects: (y, vel, target)
            force, _, _ = controller.get_action(ball.y, ball.velocity, target, return_trajectory=False)
            
        # elif controller_name == "PPO":
        #     # PPO logic needs strictly to follow the env structure usually, 
        #     # but for consistency in noise injection, we are simulating the physics manually here.
        #     # We construct the observation manually: [y, dy, target] (normalized usually in env)
        #     # Warning: This assumes the PPO was trained on raw observations or handled inside agent.
        #     # If PPOEnv normalizes, this might degrade performance.
        #     obs = np.array([ball.y, ball.velocity, target], dtype=np.float32)
        #     action, _ = controller.predict(obs, deterministic=True)
        #     force = float(action[0])
        
        # 2. Calculate Noise Manually (so we can log it)
        noise_time_cursor += config.TIME_STEP * config.PERLIN_FREQUENCY
        noise_val = generate_noise_step(noise_time_cursor, octave, base_seed)
        
        # 3. Apply Physics
        # We pass disturbance=False because we add the noise manually to the force
        # to ensure the physics engine sees (Force + Noise)
        ball.apply_force(force + noise_val, disturbance=False)
        
        # 4. Record Data
        # Format: Time, Target, Height, Velocity, Input_Force, Noise
        row = [
            round(current_time, 4),
            target,
            round(ball.y, 4),
            round(ball.velocity, 4),
            round(force, 4),
            round(noise_val, 4)
        ]
        data_rows.append(row)
        
        # 5. Accumulate Squared Error (for RMSE)
        total_sq_error += (target - ball.y) ** 2

    rmse = math.sqrt(total_sq_error / STEPS)
    return data_rows, rmse

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'Target', 'Height', 'Velocity', 'Force_Input', 'Noise_Value'])
        writer.writerows(data)

def main():
    ensure_dir(OUTPUT_DIR)
    # Start benchmark timer
    start_time = time.time()
    
    # --- Initialize Controllers ---
    controllers = {}
    
    # print("Initializing P-Controller...")
    # controllers["P_Controller"] = PController(kp=0.8)
    
    print("Initializing MPC (Standard)...")
    controllers["MPC"] = MPCController()
    controllers["MPC_Stochastic"] = MPCControllerStochastic()
    controllers["MPC_Tube"] = MPCControllerTube()
    # controllers["MPC_Acados"] = MPCControllerACADOS()
    # controllers["MPC_Stochastic_Acados"] = MPCControllerStochasticAcados()
    # controllers["MPC_Tube_Acados"] = MPCControllerTubeAcados()
    
    # if PPO_AVAILABLE and os.path.exists("ppo_ball_controller.zip"):
    #     print("Initializing PPO Agent...")
    #     # We need a dummy env to load the agent
    #     temp_env = BallEnv()
    #     try:
    #         controllers["PPO"] = load_agent(temp_env)
    #         print("PPO Loaded.")
    #     except Exception as e:
    #         print(f"Failed to load PPO: {e}")
    
    # --- Main Loop ---
    summary_results = [] # To store average errors per setting

    for name, controller in controllers.items():
        print(f"\n--- Testing Controller: {name} ---")
        
        for octave in OCTAVES_TO_TEST:
            print(f"  > Testing Octave Level: {octave}")
            
            octave_errors = []
            
            for i in range(1, RUNS_PER_SETTING + 1):
                # Run simulation
                data, error = run_simulation(controller, name, octave, i)
                octave_errors.append(error)
                
                # Save CSV
                filename = f"{name}_Oct{octave}_Run{i}.csv"
                filepath = os.path.join(OUTPUT_DIR, filename)
                save_to_csv(filepath, data)
                
            # Calculate mean RMSE for this octave
            mean_octave_error = sum(octave_errors) / len(octave_errors)
            summary_results.append({
                "Controller": name,
                "Octave": octave,
                "Avg_RMSE": mean_octave_error
            })
            print(f"    Finished 10 runs. Avg Discrepancy (RMSE): {mean_octave_error:.4f}")

    # --- Save Summary ---
    # Stop benchmark timer and compute total time
    total_time = time.time() - start_time

    summary_path = os.path.join(OUTPUT_DIR, "summary_report.csv")
    with open(summary_path, 'w', newline='') as f:
        fieldnames = ["Controller", "Octave", "Avg_RMSE", "Total_Benchmark_Time_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Write all summary rows (Total_Benchmark_Time_s will be empty for these rows)
        writer.writerows(summary_results)
        # Append a final row with the total benchmark time (in seconds)
        writer.writerow({
            "Controller": "TOTAL",
            "Octave": "",
            "Avg_RMSE": "",
            "Total_Benchmark_Time_s": round(total_time, 4)
        })

    print(f"\nTests Complete. Detailed CSVs and 'summary_report.csv' saved in '{OUTPUT_DIR}/'.")
    print(f"Total benchmark time: {total_time:.4f} seconds")

if __name__ == "__main__":
    main()