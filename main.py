import numpy as np
import pygame
import time
import os
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import cProfile
import pstats


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
from mpc_acados.acados_mpc_controller import AcadosMPCController
from filter import *
from logger import SimulationLogger
import datetime
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


def get_target_height(t: float):
    """Return the target height at time t according to config settings.

    Uses `config.MOVING_REFERENCE` and `config.MOVING_REFERENCE_TYPE` to
    choose between a sinusoidal or sigmoid reference. If moving reference
    is disabled, returns the fixed `config.TARGET_HEIGHT`.
    """
    if config.MOVING_REFERENCE:
        if config.MOVING_REFERENCE_TYPE == 'sine':
            return np.sin(t * config.SINE_REFERENCE_PERIOD) * config.SINE_REFERENCE_AMPLITUDE + config.TARGET_HEIGHT
        elif config.MOVING_REFERENCE_TYPE == 'sigmoid':
            L1 = config.TARGET_HEIGHT - config.SIGMOID_REFERENCE_AMPLITUDE
            L2 = config.TARGET_HEIGHT + config.SIGMOID_REFERENCE_AMPLITUDE - L1
            return L1 / (1 + np.exp(np.sin(config.SIGMOID_REFERENCE_PERIOD * (t - config.SIGMOID_REFERENCE_SHIFT)) * (-config.SIGMOID_REFERENCE_SLOPE))) + L2
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


def plot_data(file_path, title):
    """Loads data from an .npz file and plots it."""
    data = np.load(file_path)
    positions = data["positions"]
    forces = data["forces"]
    
    plt.figure(figsize=(12, 8))
    plt.suptitle(title)
    
    plt.subplot(2, 1, 1)
    plt.plot(positions, label='Ball Height')
    # Handle both constant and varying reference signals
    ref_data = data['ref']
    if ref_data.ndim == 0: # It's a scalar
        plt.plot([ref_data]*len(positions), 'r--', label='Target Height')
    else: # It's an array
        plt.plot(ref_data, 'r--', label='Target Height')
    plt.ylabel('Height (pixels)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(forces, label='Control Input (Force)', color='orange')
    plt.xlabel('Timestep')
    plt.ylabel('Force')
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_p_controller_sim():
    """Runs the simulation with the P-Controller."""
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Ball Simulator - P Controller")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)

    ball = Ball(config.SCREEN_WIDTH / 2, config.GROUND_HEIGHT + config.BALL_RADIUS)
    p_controller = PController(kp=0.8) # You can tune this Kp value

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        force = p_controller.get_action(ball.y, config.TARGET_HEIGHT, ball.velocity)
        ball.apply_force(force)

        # --- Drawing ---
        draw_scene(screen, font, ball, config.TARGET_HEIGHT, force)

        pygame.display.flip()
        clock.tick(1 / config.TIME_STEP)

    pygame.quit()

def run_ppo_controller_sim():
    """Runs the simulation with the RL PPO Controller."""
    agent = None
    should_train = False

    # Decide whether to train a new model or load an existing one
    if os.path.exists("ppo_ball_controller.zip"):
        train_choice = input("A pre-trained model was found. Do you want to (L)oad it or (T)rain a new one? [L/T]: ").lower()
        if train_choice == 't':
            should_train = True
    else:
        print("No pre-trained model found. Training a new one.")
        should_train = True

    if should_train:
        # --- TRAINING PHASE (NO VISUALS) ---
        print("\n--- Setting up headless environment for training... ---")
        # Create a vectorized environment for fast, parallel training
        agent = create_ppo_agent(BallEnv)
        agent = train_agent(agent, timesteps=config.PPO_TIMESTEPS)
        print("\n--- Training complete. Initializing visualization for evaluation... ---")

    # --- EVALUATION PHASE (WITH VISUALS) ---
    # Create a new environment with rendering enabled to see the results
    eval_env = BallEnv(render_mode='human')

    # Load the agent if it wasn't trained in this session
    if not should_train:
        agent = load_agent(eval_env)

    # Test the trained agent in a continuous loop
    obs, _ = eval_env.reset()
    running = True
    font = pygame.font.Font(None, 30)

    positions = []
    forces = []
    references = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)

        # --- Info Text Overlay ---
        height = obs[0]
        velocity = obs[1]
        force = action[0]
        current_ref = obs[2] # Target height is part of the observation

        positions.append(height)
        forces.append(force)
        references.append(current_ref)

        height_text = font.render(f'Height: {height:.2f}', True, config.BLACK)
        velocity_text = font.render(f'Velocity: {velocity:.2f}', True, config.BLACK)
        force_text = font.render(f'Force: {force:.2f}', True, config.BLACK)

        if eval_env.screen:
            eval_env.screen.blit(height_text, (10, 10))
            eval_env.screen.blit(velocity_text, (10, 40))
            eval_env.screen.blit(force_text, (10, 70))
            pygame.display.flip()

         # Stop the simulation if the episode ends, but continue logging
        if terminated or truncated:
            running = False
            print("Episode finished.")

    eval_env.close()
    np.savez("ppo_data.npz", positions=positions, forces=forces, ref=references)
    # Plotting the results automatically
    print("Plotting PPO simulation data...")
    plot_data("ppo_data.npz", "PPO Controller Performance")


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
        
def run_mpc_controller_ACADOS_sim(estimator: int):
    """Runs the simulation with the MPC Controller (Acados version)."""
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption(f"Ball Simulator - Acados MPC (Estimator {estimator})")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)

    ball = Ball(config.SCREEN_WIDTH / 2, config.STARTING_HEIGHT)
    
    if not os.path.exists("sim_results"):
        os.makedirs("sim_results")
    
    # Initialize Acados MPC
    # Note: This may trigger a C-code compilation step on the first run.
    #mpc_controller_acados = MPCControllerACADOS() # The old acados MPC controller
    mpc_controller_acados = AcadosMPCController()

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
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = SimulationLogger(f"sim_results/MPC_Standard_ACADOS_{timestamp}")
    
    current_step = 0
    u_prev = 0.0 #Initialization of previous input
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        current_step += 1
        t = current_step * config.TIME_STEP
        # --- State estimation ---
        est_pos, est_vel, measurements = estimate_state(estimator, ball, forces, measurements, ekf=locals().get('ekf', None), mhe=locals().get('mhe', None))

        # --- Reference Generation ---
        # Compute current target height (centralized helper)
        current_target_height = get_target_height(t)

        # --- MPC Control Step ---
        # Returns: force (float), pred_X (numpy array), pred_U (numpy array)
        force, pred_X, pred_U = mpc_controller_acados.get_action(est_pos, est_vel, u_prev, current_target_height)
        # Ensure force is a standard float for PyGame/Physics
        force = float(force)
        u_prev = force

        # Apply control
        applied_noise = ball.apply_force(force, disturbance=True)
        
        logger.log(
            time=t,
            position=ball.y,
            velocity=ball.velocity,
            target=current_target_height,
            force=force,
            noise=applied_noise,
            estimated_pos=est_pos # Valgfritt
        )

        # Logging
        positions.append(ball.y)
        velocities.append(ball.velocity)
        forces.append(force)

        # Store trajectories (convert numpy arrays to lists for serialization)
        if pred_X is not None:
            predicted_trajectories.append(pred_X.tolist())
        else:
            predicted_trajectories.append(None)

        if pred_U is not None:
            predicted_controls.append(pred_U.flatten().tolist())
        else:
            predicted_controls.append(None)

        # --- Drawing ---
        # Use labels with units for ACADOS visualization
        draw_scene(screen, font, ball, current_target_height, force,
               height_label='Height (m)', vel_label='Velocity (m/s)', force_label='Force (N)')

        pygame.display.flip()
        clock.tick(1 / config.TIME_STEP)

    pygame.quit()

    # --- Save Data ---
    # Retrieve sizes from the Acados controller
    # Note: ensure naming matches your analysis script expectations (qx vs qh)
    qh, qv, lbu, ubu, r, delta_u_max = mpc_controller_acados.sizes()
    ref = config.TARGET_HEIGHT
    
    print("Saving MPC data...")
    save_dict = {
        "positions": positions,
        "forces": forces,
        "trajectories": predicted_trajectories,
        "controls": predicted_controls,
        "N": config.ACADOS_MPC_HORIZON,
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
                
    if estimator in (3, 4) and mhe is not None:
        # Assuming MHE object has x_ests attribute
        est_states = getattr(mhe, 'x_ests', [])
        np.savez("mhe_data.npz", 
                ground_truth=[positions, velocities],
                measurements=measurements,
                estimated_states=est_states)
    
    # Lagre data og spesifikke parametere
    logger.save(extra_params={
        "Controller_Type": "Standard MPC ACADOS",
        "MPC_N": config.ACADOS_MPC_HORIZON,
        "Q_h": config.ACADOS_MPC_QH,
        "Q_v": config.ACADOS_MPC_QV,
        "R": config.ACADOS_MPC_R,
        "Estimator_Type": estimator,
        "Trajectory_Type": config.MOVING_REFERENCE_TYPE if config.MOVING_REFERENCE else config.TARGET_HEIGHT,
        "Sigmoid_Slope": config.SIGMOID_REFERENCE_SLOPE if config.MOVING_REFERENCE_TYPE == 'sigmoid' else None,
        "Sinusoidal_Period": config.SINE_REFERENCE_PERIOD if config.MOVING_REFERENCE_TYPE == 'sine' else None
    })
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
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption(f"Ball Simulator - Acados Tube MPC (Estimator {estimator})")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)

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

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
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

        # --- Drawing ---
        screen.fill(config.WHITE)
        pygame.draw.line(screen, config.BLACK, (0, config.SCREEN_HEIGHT - config.GROUND_HEIGHT), (config.SCREEN_WIDTH, config.SCREEN_HEIGHT - config.GROUND_HEIGHT), 2)
        target_y_screen = config.SCREEN_HEIGHT - current_target_height
        pygame.draw.line(screen, config.GREEN, (0, target_y_screen), (config.SCREEN_WIDTH, target_y_screen), 2)
        target_text = font.render('Target Height', True, config.GREEN)
        screen.blit(target_text, (5, target_y_screen - 25))
        ball.draw(screen)

        height_text = font.render(f'Height: {ball.y:.2f} m', True, config.BLACK)
        vel_text = font.render(f'Velocity: {ball.velocity:.2f} m/s', True, config.BLACK)
        force_text = font.render(f'Force: {force:.2f} N', True, config.BLACK)
        screen.blit(height_text, (10, 10))
        screen.blit(vel_text, (10, 40))
        screen.blit(force_text, (10, 70))

        pygame.display.flip()
        clock.tick(1 / config.TIME_STEP)

    pygame.quit()

    # --- Save Data ---
    # Retrieve sizes from the Acados controller
    qh, qv, lbu, ubu, r, delta_u_max = mpc_controller.sizes()
    ref = config.TARGET_HEIGHT
    
    print("Saving Tube MPC data...")
    save_dict = {
        "positions": positions,
        "forces": forces,
        "trajectories": predicted_trajectories,
        "controls": predicted_controls,
        "N": config.TUBE_MPC_HORIZON, # Note: Using Tube horizon
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
                
    if estimator in (3, 4) and mhe is not None:
        est_states = getattr(mhe, 'x_ests', [])
        np.savez("mhe_data.npz", 
                ground_truth=[positions, velocities],
                measurements=measurements,
                estimated_states=est_states)


if __name__ == '__main__':
    env = BallEnv()
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")
    print("Choose controller type:")
    print("1: P-Controller")
    print("2: RL PPO Controller")
    print("3: Standard MPC controller")
    print("4: Stochastic MPC controller")
    print("5: Tube MPC controller")
    print("6: Acados Standard MPC controller")
    print("7: Acados Stochastic MPC controller")
    print("8: Acados Tube MPC controller")
    choice = input("Enter choice (1-8): ")

    if choice == '1':
        run_p_controller_sim()
    elif choice == '2':
        run_ppo_controller_sim()
    elif choice == '3':
        print("Choose estimator type:")
        print("1: none (use ground truth)")
        print("2: Extended Kalman filter")
        print("3: Moving horizon estimator")
        print("4: Moving horizon estimator with acados")
        try:
            estimator_choice = int(input("enter choice (1, 2, 3 or 4): "))
        except ValueError:
            print("Invalid estimator choice: not an integer. Exiting")
        else:
            if estimator_choice in (1, 2, 3, 4):
                run_mpc_controller_sim(estimator_choice)
            else:
                print("Invalid estimator choice. Exiting")
    elif choice == '4':
        print("Choose estimator type:")
        print("1: none (use ground truth)")
        print("2: Extended Kalman filter")
        print("3: Moving horizon estimator")
        print("4: Moving horizon estimator with acados")
        try:
            estimator_choice = int(input("enter choice (1, 2, 3 or 4): "))
        except ValueError:
            print("Invalid estimator choice: not an integer. Exiting")
        else:
            if estimator_choice in (1, 2, 3, 4):
                run_mpc_controller_stochastic_sim(estimator_choice)
            else:
                print("Invalid estimator choice. Exiting")
    elif choice == '5':
        print("Choose estimator type:")
        print("1: none (use ground truth)")
        print("2: Extended Kalman filter")
        print("3: Moving horizon estimator")
        print("4: Moving horizon estimator with acados")
        try:
            estimator_choice = int(input("enter choice (1, 2, 3 or 4): "))
        except ValueError:
            print("Invalid estimator choice: not an integer. Exiting")
        else:
            if estimator_choice in (1, 2, 3, 4):
                run_mpc_controller_tube_sim(estimator_choice)
            else:
                print("Invalid estimator choice. Exiting")
    elif choice == '6':
        print("Choose estimator type:")
        print("1: none (use ground truth)")
        print("2: Extended Kalman filter")
        print("3: Moving horizon estimator")
        print("4: Moving horizon estimator with acados")
        try:
            estimator_choice = int(input("enter choice (1, 2, 3 or 4): "))
        except ValueError:
            print("Invalid estimator choice: not an integer. Exiting")
        else:
            if estimator_choice in (1, 2, 3, 4):
                run_mpc_controller_ACADOS_sim(estimator_choice)
            else:
                print("Invalid estimator choice. Exiting")
            
    elif choice == '7':
        print("Choose estimator type:")
        print("1: none (use ground truth)")
        print("2: Extended Kalman filter")
        print("3: Moving horizon estimator")
        print("4: Moving horizon estimator with acados")
        try:
            estimator_choice = int(input("enter choice (1, 2, 3 or 4): "))
        except ValueError:
            print("Invalid estimator choice: not an integer. Exiting")
        else:
            if estimator_choice in (1, 2, 3, 4):
                run_mpc_controller_stochastic_acados_sim(estimator_choice)
            else:
                print("Invalid estimator choice. Exiting")
            
    elif choice == '8':
        print("Choose estimator type:")
        print("1: none (use ground truth)")
        print("2: Extended Kalman filter")
        print("3: Moving horizon estimator")
        print("4: Moving horizon estimator with acados")
        try:
            estimator_choice = int(input("enter choice (1, 2, 3 or 4): "))
        except ValueError:
            print("Invalid estimator choice: not an integer. Exiting")
        else:
            if estimator_choice in (1, 2, 3, 4):
                run_mpc_controller_tube_acados_sim(estimator_choice)
            else:
                print("Invalid estimator choice. Exiting")
    else:
        print("Invalid choice. Exiting.")