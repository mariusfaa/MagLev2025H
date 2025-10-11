import pygame
import time
import os
from gymnasium.utils.env_checker import check_env

import config
from ball_simulation import Ball
from p_controller import PController
from environment import BallEnv
from ppo_agent import create_ppo_agent, train_agent, load_agent
from mpc_controller import MPCController

import numpy as np

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
    time_elapsed = 0.0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        current_target = config.target_height(time_elapsed)
        force = p_controller.get_action(ball.y, current_target, ball.velocity)
        ball.apply_force(force)

        # --- Drawing ---
        screen.fill(config.WHITE)
        pygame.draw.line(screen, config.BLACK, (0, config.SCREEN_HEIGHT - config.GROUND_HEIGHT), (config.SCREEN_WIDTH, config.SCREEN_HEIGHT - config.GROUND_HEIGHT), 2)
        pygame.draw.line(screen, config.GREEN, (0, config.SCREEN_HEIGHT - current_target), (config.SCREEN_WIDTH, config.SCREEN_HEIGHT - current_target), 2)
        target_text = font.render('Target Height', True, config.GREEN)
        screen.blit(target_text, (5, config.SCREEN_HEIGHT - current_target - 25))
        ball.draw(screen)

        # --- Info Text ---
        height_text = font.render(f'Height: {ball.y:.2f}', True, config.BLACK)
        velocity_text = font.render(f'Velocity: {ball.velocity:.2f}', True, config.BLACK)
        force_text = font.render(f'Force: {force:.2f}', True, config.BLACK)
        screen.blit(height_text, (10, 10))
        screen.blit(velocity_text, (10, 40))
        screen.blit(force_text, (10, 70))

        pygame.display.flip()
        clock.tick(1 / config.TIME_STEP)
        time_elapsed += config.TIME_STEP

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
        # Create a non-rendering environment for fast training
        train_env = BallEnv()
        agent = create_ppo_agent(train_env)
        agent = train_agent(agent, timesteps=config.PPO_TIMESTEPS)
        train_env.close()
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

        height_text = font.render(f'Height: {height:.2f}', True, config.BLACK)
        velocity_text = font.render(f'Velocity: {velocity:.2f}', True, config.BLACK)
        force_text = font.render(f'Force: {force:.2f}', True, config.BLACK)

        if eval_env.screen:
            eval_env.screen.blit(height_text, (10, 10))
            eval_env.screen.blit(velocity_text, (10, 40))
            eval_env.screen.blit(force_text, (10, 70))
            pygame.display.flip()

        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, _ = eval_env.reset()
            time.sleep(1)

    eval_env.close()

def run_mpc_controller_sim():
    """Runs the simulation with the MPC Controller."""
    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("Ball Simulator - MPC Controller")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)
    
    N = 25

    ball = Ball(config.SCREEN_WIDTH / 2, config.GROUND_HEIGHT + config.BALL_RADIUS)
    mpc_controller = MPCController(N, dt=config.TIME_STEP) # You can tune N and dt values
    
    positions = []
    forces = []
    predicted_trajectories = []
    predicted_controls = []

    running = True
    time_elapsed = 0.0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        current_target = config.target_height(time_elapsed)
        force, pred_X, pred_U = mpc_controller.get_action(ball.y, ball.velocity, target_height=current_target)
        # apply first control
        ball.apply_force(force, disturbance=True)

        positions.append(ball.y)
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
        screen.fill(config.WHITE)
        pygame.draw.line(screen, config.BLACK, (0, config.SCREEN_HEIGHT - config.GROUND_HEIGHT), (config.SCREEN_WIDTH, config.SCREEN_HEIGHT - config.GROUND_HEIGHT), 2)
        pygame.draw.line(screen, config.GREEN, (0, config.SCREEN_HEIGHT - current_target), (config.SCREEN_WIDTH, config.SCREEN_HEIGHT - current_target), 2)
        target_text = font.render('Target Height', True, config.GREEN)
        screen.blit(target_text, (5, config.SCREEN_HEIGHT - current_target - 25))
        ball.draw(screen)

        # --- Info Text ---
        height_text = font.render(f'Height: {ball.y:.2f}', True, config.BLACK)
        velocity_text = font.render(f'Velocity: {ball.velocity:.2f}', True, config.BLACK)
        force_text = font.render(f'Force: {force:.2f}', True, config.BLACK)
        screen.blit(height_text, (10, 10))
        screen.blit(velocity_text, (10, 40))
        screen.blit(force_text, (10, 70))

        pygame.display.flip()
        clock.tick(1 / config.TIME_STEP)
        time_elapsed += config.TIME_STEP

    pygame.quit()
    qx, qu, lbu, ubu, r, delta_u_max = mpc_controller.sizes()
    ref = config.TARGET_MEAN
    np.savez("mpc_data.npz", positions=positions, forces=forces, trajectories=predicted_trajectories, controls=predicted_controls, N=N, qx=qx, qu=qu, lbu=lbu, ubu=ubu, r=r, ref=ref, delta_u_max=delta_u_max)


if __name__ == '__main__':
    env = BallEnv()
    try:
        check_env(env)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")
    print("Choose controller type:")
    print("1: P-Controller")
    print("2: RL PPO Controller")
    print("3: MPC controller")
    choice = input("Enter choice (1, 2 or 3): ")

    if choice == '1':
        run_p_controller_sim()
    elif choice == '2':
        run_ppo_controller_sim()
    elif choice == '3':
        run_mpc_controller_sim()
    else:
        print("Invalid choice. Exiting.")
