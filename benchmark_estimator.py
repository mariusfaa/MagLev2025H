import numpy as np
import time
import datetime
from tqdm import tqdm
import pandas as pd
import os

import config
from ball_simulation import Ball
from logger import SimulationLogger

# Import Controllers
from mpc_controller import MPCController
from mpc_controller_stoch import MPCControllerStochastic
from mpc_controller_tube import MPCControllerTube

# Import Estimator Classes and Utils
# We import dynamic_model and sensor_model to initialize MHE_acados manually
from filter import run_ekf, add_noise, EKF, MHE_acados, dynamic_model, sensor_model

# --- Helper Functions ---

def get_target_height(step_idx, ref_type):
    """Calculates reference height based on config and step."""
    t = step_idx * config.TIME_STEP
    
    if ref_type == 'sine':
        return np.sin(t * config.SINE_REFERENCE_PERIOD) * config.SINE_REFERENCE_AMPLITUDE + config.TARGET_HEIGHT
    
    elif ref_type == 'sigmoid':
        L1 = config.TARGET_HEIGHT - config.SIGMOID_REFERENCE_AMPLITUDE
        L2 = config.TARGET_HEIGHT + config.SIGMOID_REFERENCE_AMPLITUDE - L1
        sigmoid_part = 1 + np.exp(np.sin(config.SIGMOID_REFERENCE_PERIOD * (t - config.SIGMOID_REFERENCE_SHIFT)) * (-config.SIGMOID_REFERENCE_SLOPE))
        return (L1 / sigmoid_part) + L2
    
    return config.TARGET_HEIGHT

def init_estimator_local(observer_type):
    """
    Initializes EKF or MHE_acados locally to ensure correct parameters.
    """
    # Lambda definitions matching filter.py
    R = lambda v1, v2: np.diag([v1, v2])
    Q = lambda v1, v2, dt: np.array([
        [v1*v2, 0.5*(v1 + v2)*dt**2],
        [0, v2*dt]
        ])
    
    # EKF (Observer 2)
    if observer_type == "EKF":
        dyn = dynamic_model(Q(config.EKF_VAR_PROC_POS, config.EKF_VAR_PROC_VEL, config.TIME_STEP))
        sens = sensor_model(R(config.EKF_VAR_MEAS_POS, config.EKF_VAR_MEAS_VEL))
        return EKF(dyn, sens)
    
    # MHE_acados (Observer 4)
    if observer_type == "MHE_acados":
        dyn = dynamic_model(Q(config.MHE_VAR_PROC_POS, config.MHE_VAR_PROC_VEL, config.TIME_STEP))
        sens = sensor_model(R(config.MHE_VAR_MEAS_POS, config.MHE_VAR_MEAS_VEL))
        Q0 = np.diag([config.STD_POS**2, config.STD_VEL**2]) # Prior covariance
        
        return MHE_acados(dyn, sens, config.MHE_HORIZON, Q0)
        
    raise ValueError(f"Unknown estimator type: {observer_type}")

def run_simulation_with_estimator(
        controller_base_name, 
        controller, 
        estimator_name, 
        ref_type, 
        param_name, 
        param_value, 
        base_folder, 
        duration=30.0
    ):
    """
    Runs a single simulation using an estimator (EKF or MHE_acados) and saves the result.
    """
    
    # Initialize Ball (Ground Truth)
    ball = Ball(config.SCREEN_WIDTH / 2, config.STARTING_HEIGHT)
    
    # Initialize Estimator
    estimator = init_estimator_local(estimator_name)

    # Prepare Filenames and Folders
    # Scenario Folder: e.g., sine_Period_1
    scenario_folder = f"{ref_type}_{param_name}_{param_value}"
    
    # Controller Name for file: e.g., MPC_Standard_MHE_acados
    full_controller_name = f"{controller_base_name}_{estimator_name}"
    
    filename = f"{base_folder}/{scenario_folder}/{full_controller_name}"
    
    logger = SimulationLogger(filename)

    # Simulation Loop
    steps = int(duration / config.TIME_STEP)
    odometry = [] # To store control inputs for the estimator
    error_squared_sum = 0.0

    # Initial inputs
    force = 0.0
    
    for i in range(steps):
        t = i * config.TIME_STEP
        target = get_target_height(i, ref_type)
        
        # 1. Measurement Step
        # Generate noisy measurement z from ground truth
        z = add_noise(ball.y, ball.velocity)
        
        # 2. Estimation Step
        if estimator_name == "EKF":
            est_pos, est_vel = run_ekf(estimator, z, odometry)
        elif estimator_name == "MHE_acados":
            # MHE_acados.run_mhe takes (meas, odometry) and returns state vector
            x_est = estimator.run_mhe(z, odometry)
            est_pos = float(x_est[0])
            est_vel = float(x_est[1])
        
        # 3. Control Step (Using Estimated State)
        # MPC controllers return: force, predicted_X, predicted_U
        f, _, _ = controller.get_action(est_pos, est_vel, target)
        force = float(f)
        
        # Store input for next estimation step
        odometry.append(force) 

        # 4. Physics Step (Ground Truth)
        noise_val = ball.apply_force(force, disturbance=True)
        
        # 5. Logging & Metrics
        error = target - ball.y
        error_squared_sum += error ** 2
        
        logger.log(
            time=t,
            position=ball.y,        # Ground Truth
            velocity=ball.velocity, # Ground Truth
            target=target,
            force=force,
            noise=noise_val,
            estimated_pos=est_pos   # Log estimated position for comparison
        )

    logger.save(extra_params={
        "Controller": controller_base_name,
        "Estimator": estimator_name,
        "Reference_Type": ref_type,
        "Varying_Param": param_name,
        "Param_Value": param_value
    })

    rmse = np.sqrt(error_squared_sum / steps)
    return rmse

def main():
    print("Starting Benchmark: Controllers with Estimators (EKF & MHE_acados)...")

    # --- SETUP TIMESTAMP FOLDER ---
    batch_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = f"benchmark_results/{batch_timestamp}"
    print(f"Saving all results to: {base_folder}")
    
    config.MOVING_REFERENCE = True
    
    # Define Controllers to Test
    controllers_to_test = [
        ("MPC_Standard", lambda: MPCController()),
        ("MPC_Stochastic", lambda: MPCControllerStochastic()),
        ("MPC_Tube", lambda: MPCControllerTube()),
    ]
    
    # Define Scenarios: (Ref Type, Param Name, Value, Config Override Key)
    # Note: We must manually set the config before running the scenario
    scenarios = [
        ("sine", "Period", 1, "SINE_REFERENCE_PERIOD"),
        ("sigmoid", "Slope", 10, "SIGMOID_REFERENCE_SLOPE")
    ]
    
    # Define Estimators order
    estimators = ["EKF", "MHE_acados"]
    
    benchmark_summary = []
    
    # --- MAIN LOOPS ---
    # Order: Estimator -> Scenario -> Controller
    
    for est_name in estimators:
        print(f"\n=== Running Benchmarks for Estimator: {est_name} ===")
        
        for ref_type, param_name, param_val, config_key in scenarios:
            print(f"  > Scenario: {ref_type} {param_name}={param_val}")
            
            # Apply Config Override
            original_config_val = getattr(config, config_key)
            setattr(config, config_key, param_val)
            
            # Update Moving Reference Type in Config
            config.MOVING_REFERENCE_TYPE = ref_type
            
            for ctrl_name, factory in tqdm(controllers_to_test, desc=f"    {est_name} / {ref_type}"):
                try:
                    # Instantiate fresh controller
                    ctrl = factory()
                    
                    rmse = run_simulation_with_estimator(
                        controller_base_name=ctrl_name,
                        controller=ctrl,
                        estimator_name=est_name,
                        ref_type=ref_type,
                        param_name=param_name,
                        param_value=param_val,
                        base_folder=base_folder
                    )
                    
                    benchmark_summary.append({
                        "Controller": ctrl_name,
                        "Estimator": est_name,
                        "Reference": ref_type,
                        "Parameter": param_val,
                        "RMSE": rmse
                    })
                    
                except Exception as e:
                    print(f"    !!! Failed: {ctrl_name} + {est_name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Restore Config
            setattr(config, config_key, original_config_val)

    # --- SAVE SUMMARY ---
    print(f"\nBenchmark finished! Folder: {base_folder}")
    summary_df = pd.DataFrame(benchmark_summary)
    summary_path = f"{base_folder}/summary_estimators_results.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()