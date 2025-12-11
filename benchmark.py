import numpy as np
import time
import os
import datetime
from tqdm import tqdm
import pandas as pd

import config
from ball_simulation import Ball
from logger import SimulationLogger
from environment import BallEnv

# Importer kontrollere
from p_controller import PController
from mpc_controller import MPCController
from mpc_controller_stoch import MPCControllerStochastic
from mpc_controller_tube import MPCControllerTube
from mpc_controller_acados import MPCControllerACADOS
from ppo_agent import load_agent

def get_target_height(step_idx, ref_type):
    """Beregner referansehøyde basert på config og steg."""
    t = step_idx * config.TIME_STEP
    
    if ref_type == 'sine':
        return np.sin(t * config.SINE_REFERENCE_PERIOD) * config.SINE_REFERENCE_AMPLITUDE + config.TARGET_HEIGHT
    
    elif ref_type == 'sigmoid':
        L1 = config.TARGET_HEIGHT - config.SIGMOID_REFERENCE_AMPLITUDE
        L2 = config.TARGET_HEIGHT + config.SIGMOID_REFERENCE_AMPLITUDE - L1
        sigmoid_part = 1 + np.exp(np.sin(config.SIGMOID_REFERENCE_PERIOD * (t - config.SIGMOID_REFERENCE_SHIFT)) * (-config.SIGMOID_REFERENCE_SLOPE))
        return (L1 / sigmoid_part) + L2
    
    return config.TARGET_HEIGHT

def calculate_complexity(ref_type, param_name, param_value):
    """Calculates the acceleration demand ratio (Complexity Score)."""
    duration = 30.0
    t_eval = np.arange(0, duration, config.TIME_STEP)
    
    # Temporarily override config to match the simulation parameters
    original_slope = config.SIGMOID_REFERENCE_SLOPE
    original_period = config.SINE_REFERENCE_PERIOD
    
    if ref_type == 'sigmoid' and param_name == 'Slope':
        config.SIGMOID_REFERENCE_SLOPE = param_value
    elif ref_type == 'sine' and param_name == 'Period':
        config.SINE_REFERENCE_PERIOD = param_value

    # Generate target trajectory
    targets = np.array([get_target_height(i, ref_type) for i in range(len(t_eval))])

    # Restore config
    config.SIGMOID_REFERENCE_SLOPE = original_slope
    config.SINE_REFERENCE_PERIOD = original_period

    # Calculate max acceleration demand
    velocity_ref = np.gradient(targets, config.TIME_STEP)
    accel_ref = np.gradient(velocity_ref, config.TIME_STEP)
    max_ref_accel = np.max(np.abs(accel_ref))

    # Calculate available acceleration (Force/Mass - Gravity)
    max_actuator_accel = config.FORCE_MAGNITUDE / config.BALL_MASS
    available_accel = max_actuator_accel - config.GRAVITY
    
    # Add noise floor if strictly needed, otherwise just return ratio
    return max_ref_accel / available_accel

def run_single_simulation(controller_name, controller, ref_type, param_name, param_value, base_folder, duration=30.0):
    """Kjører en simulering og lagrer til spesifisert mappe."""
    
    ball = Ball(config.SCREEN_WIDTH / 2, config.STARTING_HEIGHT)
    
    # Struktur: benchmark_results/<TIMESTAMP>/sigmoid_Slope_5/<Controller>.csv
    scenario_folder = f"{ref_type}_{param_name}_{param_value}"
    # Her fjerner vi timestamp fra selve filnavnet for å gjøre LaTeX enklere
    filename = f"{base_folder}/{scenario_folder}/{controller_name}"
    
    logger = SimulationLogger(filename)

    error_squared_sum = 0.0  # <--- Initialize error tracker
    steps = int(duration / config.TIME_STEP)
    
    for i in range(steps):
        t = i * config.TIME_STEP
        target = get_target_height(i, ref_type)
        
        force = 0.0
        
        # if controller_name == "PPO":
        #     obs = np.array([ball.y, ball.velocity, target], dtype=np.float32)
        #     action, _ = controller.predict(obs, deterministic=True)
        #     force = float(action[0])
            
        # if controller_name == "P-Controller":
        #     force = controller.get_action(ball.y, target, ball.velocity)
            
        if "MPC" in controller_name:
            f, _, _ = controller.get_action(ball.y, ball.velocity, target)
            force = float(f)

        noise = ball.apply_force(force, disturbance=True)
        error = target - ball.y
        error_squared_sum += error ** 2
        
        logger.log(
            time=t,
            position=ball.y,
            velocity=ball.velocity,
            target=target,
            force=force,
            noise=noise,
            estimated_pos=ball.y
        )

    logger.save(extra_params={
        "Controller": controller_name,
        "Reference_Type": ref_type,
        "Varying_Param": param_name,
        "Param_Value": param_value
    })

    rmse = np.sqrt(error_squared_sum / steps)
    return rmse
    
def main():
    print("Starter Benchmark av kontrollere...")

    # --- SETUP TIMESTAMP FOLDER ---
    # Lager én timestamp for hele kjøringen
    batch_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = f"benchmark_results/{batch_timestamp}"
    print(f"Lagrer alle resultater i: {base_folder}")
    
    config.MOVING_REFERENCE = True
    
    # print("Laster PPO modell...")
    # try:
    #     dummy_env = BallEnv(render_mode=None)
    #     ppo_agent = load_agent(dummy_env)
    # except Exception as e:
    #     print(f"Kunne ikke laste PPO: {e}. Hopper over PPO.")
    #     ppo_agent = None

    controllers_to_test = [
        ("MPC_Standard", lambda: MPCController()),
        ("MPC_Stochastic", lambda: MPCControllerStochastic()),
        ("MPC_Tube", lambda: MPCControllerTube()),
        # ("MPC_Acados", lambda: MPCControllerACADOS()), 
    ]
    
    benchmark_summary = []
    
    # 1. Sigmoid Slope Test
    slopes = [5, 10, 15]
    config.MOVING_REFERENCE_TYPE = 'sigmoid'
    
    print("\n--- Kjører Sigmoid Slope Tester ---")
    for slope in slopes:
        config.SIGMOID_REFERENCE_SLOPE = slope
        complexity = calculate_complexity('sigmoid', 'Slope', slope)
        for name, factory in tqdm(controllers_to_test, desc=f"Slope {slope}"):
            try:
                ctrl = factory()
                rmse = run_single_simulation(name, ctrl, 'sigmoid', 'Slope', slope, base_folder)
                
                benchmark_summary.append({
                    "Controller": name,
                    "Reference": "Sigmoid",
                    "Parameter": slope,
                    "Complexity": complexity,
                    "RMSE": rmse
                })
            except Exception as e:
                print(f"Feilet for {name}: {e}")

    # 2. Sinusoidal Period Test
    periods = [0.1, 1, 10] 
    config.MOVING_REFERENCE_TYPE = 'sine'
    
    print("\n--- Kjører Sinusoidal Period Tester ---")
    for period in periods:
        config.SINE_REFERENCE_PERIOD = period
        complexity = calculate_complexity('sine', 'Period', period)
        for name, factory in tqdm(controllers_to_test, desc=f"Period {period}"):
            try:
                ctrl = factory()
                rmse = run_single_simulation(name, ctrl, 'sine', 'Period', period, base_folder)
                benchmark_summary.append({
                    "Controller": name,
                    "Reference": "Sine",
                    "Parameter": period,
                    "Complexity": complexity,
                    "RMSE": rmse
                })
            except Exception as e:
                print(f"Feilet for {name}: {e}")

    print(f"\nBenchmark fullført! Mappe: {base_folder}")
    summary_df = pd.DataFrame(benchmark_summary)
    summary_path = f"{base_folder}/summary_results.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Benchmark summary saved to: {summary_path}")

if __name__ == "__main__":
    main()