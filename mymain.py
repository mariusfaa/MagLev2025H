import numpy as np
import pygame
import time
import os
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import cProfile
import pstats
import random
from noise import pnoise1


import config
from ball_simulation import Ball
from p_controller import PController
from environment import BallEnv
from ppo_agent import create_ppo_agent, train_agent, load_agent
from mpc_controller import MPCController
from mpc_controller_stoch import MPCControllerStochastic
from mpc_controller_tube import MPCControllerTube
from mpc_controller_acados import MPCControllerACADOS
from mpc_controller_stoch_acados import MPCControllerStochasticAcados
from mpc_controller_tube_acados import MPCControllerTubeAcados
from filter import *
from logger import SimulationLogger
import datetime
import pandas as pd


def generate_noise_step(time_cursor, octave, base):
    """Replicates the noise generation logic to log the exact value."""
    # Note: pnoise1 returns value roughly in [-1, 1]
    return pnoise1(time_cursor, octaves=octave, base=base) * config.PERLIN_AMPLITUDE

def draw_scene(screen, font, ball, current_target_height, force,
               height_label='Height', vel_label='Velocity', force_label='Force'):
    """Draw ground, target line, ball and info texts to the screen.

    Parameters:
    - screen, font, ball: pygame objects
    - current_target_height: numeric, height in same units as ball.y
    - force: numeric, control input to display
    - height_label, vel_label, force_label: labels used in the info overlay
    """
    screen.fill(config.WHITE)
    pygame.draw.line(screen, config.BLACK,
                     (0, config.SCREEN_HEIGHT - config.GROUND_HEIGHT),
                     (config.SCREEN_WIDTH, config.SCREEN_HEIGHT - config.GROUND_HEIGHT), 2)
    pygame.draw.line(screen, config.GREEN,
                     (0, config.SCREEN_HEIGHT - current_target_height),
                     (config.SCREEN_WIDTH, config.SCREEN_HEIGHT - current_target_height), 2)
    target_text = font.render('Target Height', True, config.GREEN)
    # place the label slightly above the target line
    screen.blit(target_text, (5, config.SCREEN_HEIGHT - current_target_height - 25))
    ball.draw(screen)

    # Info Text Overlay
    height_text = font.render(f'{height_label}: {ball.y:.2f}', True, config.BLACK)
    velocity_text = font.render(f'{vel_label}: {ball.velocity:.2f}', True, config.BLACK)
    force_text = font.render(f'{force_label}: {force:.2f}', True, config.BLACK)
    screen.blit(height_text, (10, 10))
    screen.blit(velocity_text, (10, 40))
    screen.blit(force_text, (10, 70))


def get_target_height(step_idx, ref_type, sig_slope=config.SIGMOID_REFERENCE_SLOPE, sine_freq=config.SINE_REFERENCE_PERIOD):
    """Beregner referansehøyde basert på config og steg."""
    t = step_idx * config.TIME_STEP
    
    if ref_type == 'sine':
        return np.sin(t * sine_freq) * config.SINE_REFERENCE_AMPLITUDE + config.TARGET_HEIGHT
    
    elif ref_type == 'sigmoid':
        L1 = config.TARGET_HEIGHT - config.SIGMOID_REFERENCE_AMPLITUDE
        L2 = config.TARGET_HEIGHT + config.SIGMOID_REFERENCE_AMPLITUDE - L1
        sigmoid_part = 1 + np.exp(np.sin(config.SIGMOID_REFERENCE_PERIOD * (t - config.SIGMOID_REFERENCE_SHIFT)) * (-sig_slope))
        return (L1 / sigmoid_part) + L2
    
    return config.TARGET_HEIGHT


def estimate_state(estimator: int, ball, forces, measurements, ekf=None, mhe=None):
    """Perform measurement, append to `measurements`, and run selected estimator.

    Returns (est_pos, est_vel, measurements).
    - `estimator`: int selector (1..4)
    - `ball`: Ball instance (for ground truth and to get y, velocity)
    - `forces`: list of past applied forces (passed to estimators that need it)
    - `measurements`: numpy array with shape (2, N) to append the new measurement to
    - `ekf`, `mhe`: optional estimator objects required by EKF/MHE modes
    """
    z_meas = add_noise(ball.y, ball.velocity)
    measurements = np.append(measurements, z_meas, axis=1)

    if estimator == 1:  # no estimator, use ground truth
        est_pos, est_vel = ball.y, ball.velocity
    elif estimator == 2:  # EKF
        est_pos, est_vel = run_ekf(ekf, z_meas, forces)
    elif estimator == 3:  # MHE
        est_pos, est_vel = run_mhe(mhe, z_meas, forces)
    elif estimator == 4:  # MHE acados
        est_pos, est_vel = mhe.run_mhe(z_meas, forces)
    else:
        est_pos, est_vel = ball.y, ball.velocity

    return est_pos, est_vel, measurements


def run_mpc_controller_sim(estimator: int):
    """Runs the simulation with the MPC Controller."""
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Ball Simulator - MPC Controller")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)


    ball = Ball(config.SCREEN_WIDTH / 2, config.STARTING_HEIGHT)
    mpc_controller = MPCController()

    # --- Initialize chosen estimator ---
    if estimator == 2:
        ekf = init_estimator(estimator)
    if estimator in (3, 4):
        mhe = init_estimator(estimator)
    
    positions = []
    velocities = []
    forces = []
    predicted_trajectories = []
    predicted_controls = []

    measurements = np.empty((2,0))
    
    # Lagre mappe hvis den ikke finnes
    if not os.path.exists("sim_results"):
        os.makedirs("sim_results")
    
    current_step = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        current_step += 1
        t = current_step * config.TIME_STEP

        # --- State estimation ---
        est_pos, est_vel, measurements = estimate_state(estimator, ball, forces, measurements, ekf=locals().get('ekf', None), mhe=locals().get('mhe', None))
        
        # Compute current target height (centralized helper)
        current_target_height = get_target_height(t)
        force, pred_X, pred_U = mpc_controller.get_action(est_pos, est_vel, current_target_height)
        # apply first control
        applied_noise = ball.apply_force(force, disturbance=True)
        
        positions.append(ball.y)
        velocities.append(ball.velocity)
        forces.append(force)

        # store predicted trajectory (convert to simple lists) if available
        if pred_X is not None:
            # pred_X shape (2, N+1) -> store heights and velocities separately or together
            predicted_trajectories.append(pred_X.tolist())
        else:
            predicted_trajectories.append(None)

        if pred_U is not None:
            predicted_controls.append(pred_U.flatten().tolist())
        else:
            predicted_controls.append(None)


        # --- Drawing ---
        draw_scene(screen, font, ball, current_target_height, force)

        pygame.display.flip()
        clock.tick(1 / config.TIME_STEP)

    pygame.quit()
    qx, qu, lbu, ubu, r, delta_u_max = mpc_controller.sizes()
    ref = config.TARGET_HEIGHT
    np.savez("mpc_data.npz", positions=positions, forces=forces, trajectories=predicted_trajectories, controls=predicted_controls, N=config.STD_MPC_HORIZON, qx=qx, qu=qu, lbu=lbu, ubu=ubu, r=r, ref=ref, delta_u_max=delta_u_max)

    if estimator == 2:
        np.savez("ekf_data.npz", ground_truth=[positions, velocities],
                measurements=measurements,
                estimated_states=ekf.state_ests,
                estimated_measurements=ekf.meas_ests)
    if estimator in (3, 4):
        np.savez("mhe_data.npz", ground_truth=[positions, velocities],
                measurements=measurements,
                estimated_states=mhe.x_ests)
        

def run_mpc_controller_stochastic_sim(estimator: int):
    """Runs the simulation with the stochastic MPC Controller."""
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Ball Simulator - stochastic MPC Controller")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)

    ball = Ball(config.SCREEN_WIDTH / 2, config.STARTING_HEIGHT)
    mpc_controller_stoch = MPCControllerStochastic() # You can tune N, dt and num_samples values

    # --- Initialize chosen estimator ---
    if estimator == 2:
        ekf = init_estimator(estimator)
    elif estimator in (3, 4):
        mhe = init_estimator(estimator)
    
    positions = []
    velocities = []
    forces = []
    predicted_trajectories = []
    predicted_controls = []

    measurements = np.empty((2,0))
    current_step = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        current_step += 1

        # --- State estimation ---
        est_pos, est_vel, measurements = estimate_state(estimator, ball, forces, measurements, ekf=locals().get('ekf', None), mhe=locals().get('mhe', None))

        # Compute current target height (centralized helper)
        current_target_height = get_target_height(current_step * config.TIME_STEP)
        force, pred_X, pred_U = mpc_controller_stoch.get_action(est_pos, est_vel, current_target_height)
        # apply first control
        ball.apply_force(force, disturbance=False)

        positions.append(ball.y)
        velocities.append(ball.velocity)
        forces.append(force)

        # store predicted trajectory (convert to simple lists) if available
        if pred_X is not None:
            # pred_X shape (2, N+1) -> store heights and velocities separately or together
            predicted_trajectories.append(pred_X.tolist())
        else:
            predicted_trajectories.append(None)

        if pred_U is not None:
            predicted_controls.append(pred_U.flatten().tolist())
        else:
            predicted_controls.append(None)

        # --- Drawing ---
        draw_scene(screen, font, ball, current_target_height, force)

        pygame.display.flip()
        clock.tick(1 / config.TIME_STEP)

    pygame.quit()
    qx, qu, lbu, ubu, r, delta_u_max = mpc_controller_stoch.sizes()
    ref = config.TARGET_HEIGHT
    np.savez("mpc_data.npz", positions=positions, forces=forces, trajectories=predicted_trajectories, controls=predicted_controls, N=config.STOCHASTIC_MPC_HORIZON, qx=qx, qu=qu, lbu=lbu, ubu=ubu, r=r, ref=ref, delta_u_max=delta_u_max)

    if estimator == 2:
        np.savez("ekf_data.npz", ground_truth=[positions, velocities],
                measurements=measurements,
                estimated_states=ekf.state_ests,
                estimated_measurements=ekf.meas_ests)
    elif estimator in (3, 4):
        np.savez("mhe_data.npz", ground_truth=[positions, velocities],
                measurements=measurements,
                estimated_states=mhe.x_ests)

def run_mpc_controller_tube_sim(estimator: int):
    """Runs the simulation with the Tube MPC Controller."""
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Ball Simulator - Tube MPC Controller")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)

    ball = Ball(config.SCREEN_WIDTH / 2, config.STARTING_HEIGHT)
    mpc_controller = MPCControllerTube()

    # --- Initialize chosen estimator ---
    if estimator == 2:
        ekf = init_estimator(estimator)
    elif estimator in (3, 4):
        mhe = init_estimator(estimator)

    positions = []
    velocities = []
    forces = []
    predicted_trajectories = []
    predicted_controls = []

    measurements = np.empty((2,0))
    current_step = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        current_step += 1

        # --- State estimation ---
        est_pos, est_vel, measurements = estimate_state(estimator, ball, forces, measurements, ekf=locals().get('ekf', None), mhe=locals().get('mhe', None))

        # Compute current target height (centralized helper)
        current_target_height = get_target_height(current_step * config.TIME_STEP)
        force, pred_X, pred_U = mpc_controller.get_action(est_pos, est_vel, current_target_height)
        ball.apply_force(force, disturbance=True)

        positions.append(ball.y)
        velocities.append(ball.velocity)
        forces.append(force)

        if pred_X is not None:
            predicted_trajectories.append(pred_X.tolist())
        else:
            predicted_trajectories.append(None)

        if pred_U is not None:
            predicted_controls.append(pred_U.flatten().tolist())
        else:
            predicted_controls.append(None)

        # --- Drawing ---
        draw_scene(screen, font, ball, current_target_height, force)

        pygame.display.flip()
        clock.tick(1 / config.TIME_STEP)

    pygame.quit()
    qx, qu, lbu, ubu, r, delta_u_max = mpc_controller.sizes()
    ref = config.TARGET_HEIGHT
    np.savez("mpc_data.npz", positions=positions, forces=forces, trajectories=predicted_trajectories, controls=predicted_controls, N=config.STD_MPC_HORIZON, qx=qx, qu=qu, lbu=lbu, ubu=ubu, r=r, ref=ref, delta_u_max=delta_u_max)

    if estimator == 2:
        np.savez("ekf_data.npz", ground_truth=[positions, velocities],
                measurements=measurements,
                estimated_states=ekf.state_ests,
                estimated_measurements=ekf.meas_ests)
    elif estimator in (3, 4):
        np.savez("mhe_data.npz", ground_truth=[positions, velocities],
                measurements=measurements,
                estimated_states=mhe.x_ests)
        
def run_mpc_controller_ACADOS_sim(estimator, test_type, ref_type='constant', freq=config.SINE_REFERENCE_PERIOD, slope=config.SIGMOID_REFERENCE_SLOPE, octave=config.PERLIN_OCTAVES, horizon=config.MHE_HORIZON, increasing=config.MHE_INCREASING_HORIZON, erk=config.MHE_INTEGRATOR=='ERK'):
    """Runs the simulation with the MPC Controller (Acados version)."""

    ball = Ball(config.SCREEN_WIDTH / 2, config.STARTING_HEIGHT)
    
    # Initialize Acados MPC
    # Note: This may trigger a C-code compilation step on the first run.
    mpc_controller_acados = MPCControllerACADOS()

    # --- Initialize chosen estimator ---
    if estimator == 'ekf':
        ekf = init_estimator(estimator)
    elif estimator == 'mhe_acados':
        mhe = init_estimator(estimator, horizon=horizon)
    
    positions = []
    velocities = []
    forces = []
    error_pos = []
    error_vel = []

    measurements = np.empty((2, 0))

    
    # --- Disturbance ---
    # Random base for this specific run
    base_seed = random.randint(0, 100)
    
    # Noise tracking
    noise_time_cursor = 0.0
    disturbances = []
    
    current_step = 0
    duration = 30
    max_step = duration/config.TIME_STEP
    while current_step < max_step:

        current_step += 1

        # --- State estimation ---
        z_meas = add_noise(ball.y, ball.velocity)
        measurements = np.append(measurements, z_meas, axis=1)

        if estimator == 'ekf':  # EKF
            est_pos, est_vel = run_ekf(ekf, z_meas, forces, np.array([ball.y, ball.velocity]))
        elif estimator == 'mhe_acados':  # MHE acados
            est_pos, est_vel = mhe.run_mhe(z_meas, forces)
        else:  # no estimator, use ground truth
            est_pos, est_vel = ball.y, ball.velocity

        # --- Reference Generation ---
        # Compute current target height (centralized helper)
        if ref_type == 'constant':
            current_target_height = config.TARGET_HEIGHT
        elif ref_type == 'sine':
            current_target_height = get_target_height(current_step, ref_type, sine_freq=freq)
        elif ref_type == 'sigmoid':
            current_target_height = get_target_height(current_step, ref_type, sig_slope=slope)
            

        # --- MPC Control Step ---
        # Returns: force (float), pred_X (numpy array), pred_U (numpy array)
        force, _, _ = mpc_controller_acados.get_action(est_pos, est_vel, current_target_height)
        
        # Ensure force is a standard float for PyGame/Physics
        force = float(force)

        # --- Disturbance ---
        noise_time_cursor += config.TIME_STEP * config.PERLIN_FREQUENCY
        noise_val = generate_noise_step(noise_time_cursor, octave, base_seed)

        
        # Logging
        positions.append(ball.y)
        velocities.append(ball.velocity)
        forces.append(force)
        disturbances.append(noise_val)
        error_pos.append(ball.y-est_pos)
        error_vel.append(ball.velocity-est_vel)

        # Apply control and states
        ball.apply_force(force + noise_val)



    # --- Save Data ---
    save_folder = f'filter_results/{test_type}/'
    if estimator == 'ekf':
        df_ekf = pd.DataFrame({
                'disturbance': disturbances,
                'error_pos': error_pos,
                'error_vel': error_vel,
                'pos_gt': positions,
                'vel_gt': velocities,
                'pos_meas': measurements[0],
                'vel_meas': measurements[1],
                'pos_ests': ekf.state_ests_mean[0],
                'vel_ests': ekf.state_ests_mean[1],
                'nis_values': ekf.nis_values,
                'nees_values': ekf.nees_values,
                'runtimes': ekf.runtimes
                })
        df_ekf.to_csv(f'{save_folder}ekf_ref{ref_type}_freq{freq}_slope{slope}_octave{octave}.csv', index=False)
                
    if estimator == 'mhe_acados':
        df_mhe = pd.DataFrame({
                'disturbance': disturbances,
                'error_pos': error_pos,
                'error_vel': error_vel,
                'pos_gt': positions,
                'vel_gt': velocities,
                'pos_meas': measurements[0],
                'vel_meas': measurements[1],
                'pos_ests': mhe.x_ests[0],
                'vel_ests': mhe.x_ests[1],
                'runtimes': mhe.runtimes,
                'runtimes_kalman': mhe.runtimes_kalman
                })
        df_mhe.to_csv(f'{save_folder}mhe_ref{ref_type}_freq{freq}_slope{slope}_octave{octave}_horizon{horizon}_growing{increasing}_erk{erk}.csv', index=False)
    
    
def run_mpc_controller_stochastic_acados_sim(estimator: int):
    """Runs the simulation with the Stochastic MPC Controller (Acados version)."""
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption(f"Ball Simulator - Acados Stochastic MPC (Estimator {estimator})")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)

    ball = Ball(config.SCREEN_WIDTH / 2, config.STARTING_HEIGHT)
    
    # Initialize Acados Stochastic MPC
    mpc_controller = MPCControllerStochasticAcados()

    # --- Initialize chosen estimator ---
    ekf, mhe = None, None
    if estimator == 2:
        ekf = init_estimator(estimator)
    elif estimator in (3, 4):
        mhe = init_estimator(estimator)
    
    positions = []
    velocities = []
    forces = []
    predicted_trajectories = []
    predicted_controls = []

    measurements = np.empty((2, 0))
    current_step = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        current_step += 1

        # --- State estimation ---
        est_pos, est_vel, measurements = estimate_state(estimator, ball, forces, measurements, ekf=ekf, mhe=mhe)
        
        est_pos = float(est_pos)
        est_vel = float(est_vel)

        # --- Reference Generation ---
        t = current_step * config.TIME_STEP
        current_target_height = get_target_height(t)

        # --- MPC Control Step ---
        force, pred_X, pred_U = mpc_controller.get_action(est_pos, est_vel, current_target_height)
        
        force = float(force)
        ball.apply_force(force, disturbance=True) # Disturbance usually True to test robustness

        # Logging
        positions.append(ball.y)
        velocities.append(ball.velocity)
        forces.append(force)

        if pred_X is not None:
            predicted_trajectories.append(pred_X.tolist())
        else:
            predicted_trajectories.append(None)

        if pred_U is not None:
            predicted_controls.append(pred_U.flatten().tolist())
        else:
            predicted_controls.append(None)

        # --- Drawing ---
        draw_scene(screen, font, ball, current_target_height, force,
               height_label='Height (m)', vel_label='Velocity (m/s)', force_label='Force (N)')

        pygame.display.flip()
        clock.tick(1 / config.TIME_STEP)

    pygame.quit()

    # --- Save Data ---
    qh, qv, lbu, ubu, r, delta_u_max = mpc_controller.sizes()
    ref = config.TARGET_HEIGHT
    
    print("Saving Stochastic MPC (Acados) data...")
    save_dict = {
        "positions": positions,
        "forces": forces,
        "trajectories": predicted_trajectories,
        "controls": predicted_controls,
        "N": config.STOCHASTIC_MPC_HORIZON,
        "qh": qh, 
        "qv": qv, 
        "lbu": lbu, 
        "ubu": ubu, 
        "r": r, 
        "ref": ref, 
        "delta_u_max": delta_u_max
    }
    
    np.savez("mpc_data.npz", **save_dict)

    if estimator == 2 and ekf is not None:
        np.savez("ekf_data.npz", 
                ground_truth=[positions, velocities],
                measurements=measurements,
                estimated_states=ekf.state_ests,
                estimated_measurements=ekf.meas_ests)
    elif estimator in (3, 4) and mhe is not None:
        est_states = getattr(mhe, 'x_ests', [])
        np.savez("mhe_data.npz", 
                ground_truth=[positions, velocities],
                measurements=measurements,
                estimated_states=est_states)
    
def run_mpc_controller_tube_acados_sim(estimator: int):
    """Runs the simulation with the Tube MPC Controller (Acados version)."""

    ball = Ball(config.SCREEN_WIDTH / 2, config.GROUND_HEIGHT + config.BALL_RADIUS)
    
    # Initialize Acados Tube MPC
    # Note: This may trigger a C-code compilation step on the first run.
    mpc_controller = MPCControllerTubeAcados()

    # --- Initialize chosen estimator ---
    ekf, mhe = None, None
    if estimator == 2:
        ekf = init_estimator(estimator)
    elif estimator in (3, 4):
        mhe = init_estimator(estimator)
    
    positions = []
    velocities = []
    forces = []
    predicted_trajectories = []
    predicted_controls = []

    measurements = np.empty((2, 0))
    current_step = 0
    running = True

    while current_step < 200:
        current_step += 1

        # --- State estimation ---
        z_meas = add_noise(ball.y, ball.velocity)
        measurements = np.append(measurements, z_meas, axis=1)

        if estimator == 1: # No estimator, use ground truth
            est_pos, est_vel = ball.y, ball.velocity
        elif estimator == 2: # EKF
            est_pos, est_vel = run_ekf(ekf, z_meas, forces)
        elif estimator == 3: # MHE
            est_pos, est_vel = run_mhe(mhe, z_meas, forces)
        elif estimator == 4: # MHE acados
            est_pos, est_vel = mhe.run_mhe(z_meas, forces)
        else:
            est_pos, est_vel = ball.y, ball.velocity

        # --- Reference Generation ---
        t = current_step * config.TIME_STEP
        current_target_height = get_target_height(t)

        # --- MPC Control Step ---
        # Returns: force (float), pred_X (numpy array), pred_U (numpy array)
        force, pred_X, pred_U = mpc_controller.get_action(est_pos, est_vel, current_target_height)
        
        # Ensure force is a standard float
        force = float(force)

        # Apply control (Disturbance=True is important to test the Tube robustness!)
        ball.apply_force(force, disturbance=True)

        # Logging
        positions.append(ball.y)
        velocities.append(ball.velocity)
        forces.append(force)

        if pred_X is not None:
            predicted_trajectories.append(pred_X.tolist())
        else:
            predicted_trajectories.append(None)

        if pred_U is not None:
            predicted_controls.append(pred_U.flatten().tolist())
        else:
            predicted_controls.append(None)





    # --- Save Data ---
    # Retrieve sizes from the Acados controller
    qh, qv, lbu, ubu, r, delta_u_max = mpc_controller.sizes()
    ref = config.TARGET_HEIGHT
    

    


    if estimator == 2 and ekf is not None:
        np.savez("ekf_data.npz", 
                ground_truth=[positions, velocities],
                measurements=measurements,
                estimated_states=ekf.state_ests,
                estimated_measurements=ekf.meas_ests)
                
    if estimator in (3, 4) and mhe is not None:
        est_states = getattr(mhe, 'x_ests', [])
        np.savez("mhe_data.npz", 
                ground_truth=[positions, velocities],
                measurements=measurements,
                estimated_states=est_states)


def run_benchmark():
    octaves = [2, 3, 4]
    slopes = [5, 10, 15]
    freqs = [0.1, 1, 10]
    filters = ['ekf', 'mhe_acados']
    horizons = [20, 50, 100]
    horizons_many = np.arange(0, 101, 10).tolist(); horizons_many[0] = 1  # [1, 10, ..., 100]

    # test EKF and MHE_acados
    for est in filters:

        # sinusoidal reference tests
        for freq in freqs:
            print(f'{est} freq: {freq}')
            run_mpc_controller_ACADOS_sim(est, 'sine', ref_type='sine', freq=freq)
        
        # sigmoid reference tests
        for slope in slopes:
            print(f'{est} slope: {slope}')
            run_mpc_controller_ACADOS_sim(est, 'sigmoid', ref_type='sigmoid', slope=slope)

        # disturbance tests
        for octave in octaves:
            print(f'{est} octave: {octave}')
            run_mpc_controller_ACADOS_sim(est, 'disturbance', octave=octave)
        
        # horizon tests againts high disturbance and for runtime benchmarking
        
        if est == 'mhe_acados':
            config.PERLIN_AMPLITUDE=40
            for horizon in horizons_many:
                print(f'{est} horizon: {horizon}')
                run_mpc_controller_ACADOS_sim(est, 'horizon', horizon=horizon)
            
            # growing horizons with and without ERK
            config.PERLIN_AMPLITUDE=10
            config.MHE_INTEGRATOR='ERK'
            run_mpc_controller_ACADOS_sim(est, 'increasing', erk=True)
            run_mpc_controller_ACADOS_sim(est, 'increasing', horizon=100, erk=True)
            config.MHE_INCREASING_HORIZON = True
            run_mpc_controller_ACADOS_sim(est, 'increasing', horizon=100, increasing=True, erk=True)
            config.MHE_INTEGRATOR='DISCRETE'
            run_mpc_controller_ACADOS_sim(est, 'increasing', horizon=100, increasing=True, )
            




#TODO full plots in appendix. interesting plots in results with rmse. instantaneous error plots? list table of rmse values for easy comparison in results. only show runtime when comparing horizon length

# show with erk and discrete for short and long horizon
# model error plots
if __name__ == '__main__':
    run_benchmark()


if __name__ == 'mymain':
    est = 'mhe_acados'
    config.PERLIN_AMPLITUDE=100
    run_mpc_controller_ACADOS_sim(est, octave=4)
    if est == 'ekf':
        import plot_ekf_data
    elif est == 'mhe_acados':
        import plot_mhe_data