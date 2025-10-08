import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import config
from ball_simulation import Ball

class BallEnv(gym.Env):
    """
    Custom OpenAI Gym environment for the ball simulation.
    This class defines the state and action spaces, and the logic for
    stepping through the simulation, calculating rewards, and rendering.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 50}

    def __init__(self, render_mode=None):
        super(BallEnv, self).__init__()
        self.render_mode = render_mode

        # Action: A continuous force value to apply to the ball.
        self.action_space = spaces.Box(low=-20, high=20, shape=(1,), dtype=np.float32)

        # Observation: The ball's current height and vertical velocity.
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf]),
            shape=(3,),
            dtype=np.float32
        )

        self.ball = None
        self.screen = None
        self.clock = None
        self.font = None
        self.current_step = 0

    def _get_obs(self):
        """Returns the current observation."""
        distance_to_target = config.TARGET_HEIGHT - self.ball.y
        return np.array([self.ball.y, self.ball.velocity, distance_to_target], dtype=np.float32)
    
    def _get_info(self):
        """Returns auxiliary diagnostic information."""
        return {"distance_to_target": np.abs(self.ball.y - config.TARGET_HEIGHT)}

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        # Start the ball at a random height for better training generalization
        initial_y = self.np_random.uniform(
            config.GROUND_HEIGHT + config.BALL_RADIUS,
            config.SCREEN_HEIGHT - 100
        )
        self.ball = Ball(config.SCREEN_WIDTH / 2, initial_y)
        self.current_step = 0 # Reset the step counter

        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), self._get_info()

    def step(self, action):
        """Executes one time step within the environment."""
        self.current_step += 1 # Increment step counter
        force = action[0] #+ 4.81
        self.ball.apply_force(force)

        is_out_of_bounds = (self.ball.y < config.GROUND_HEIGHT + config.BALL_RADIUS or
                        self.ball.y > config.SCREEN_HEIGHT - config.BALL_RADIUS)

        if is_out_of_bounds:
            # End the episode and give a large penalty if out of bounds
            reward = -100.0
            terminated = True
        else:
            distance = np.linalg.norm(self.ball.y - config.TARGET_HEIGHT)
            reward = 4-distance*0.01

            error = config.TARGET_HEIGHT - self.ball.y
            is_moving_away = np.sign(error) * np.sign(self.ball.velocity) < 0
            if is_moving_away:
                reward -= 1

            terminated = False

        if self.render_mode == "human":
            self._render_frame()

        truncated = self.current_step >= config.MAX_EPISODE_STEPS
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        """Renders the environment."""
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        """Helper function to draw the current state of the environment."""
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Ball Controller Simulator")
            self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

        self.screen.fill(config.WHITE)
        
        # Draw ground line
        pygame.draw.line(self.screen, config.BLACK, (0, config.SCREEN_HEIGHT - config.GROUND_HEIGHT), (config.SCREEN_WIDTH, config.SCREEN_HEIGHT - config.GROUND_HEIGHT), 2)
        
        # Draw target height line
        pygame.draw.line(self.screen, config.GREEN, (0, config.SCREEN_HEIGHT - config.TARGET_HEIGHT), (config.SCREEN_WIDTH, config.SCREEN_HEIGHT - config.TARGET_HEIGHT), 2)
        target_text = self.font.render("Target", True, config.GREEN)
        self.screen.blit(target_text, (10, config.SCREEN_HEIGHT - config.TARGET_HEIGHT - 20))

        # Draw the ball
        self.ball.draw(self.screen)
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Closes the environment and quits pygame."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
