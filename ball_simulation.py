import pygame
import config
import numpy as np
from noise import pnoise1

class Ball:
    """Represents the ball and its physics in the simulation."""
    def __init__(self, x, y):
        """Initializes the ball at a given position."""
        self.x = x
        self.y = y
        self.velocity = 0.0
        self._noise_time = 0.0

    def apply_force(self, force, disturbance=False):
        """
        Applies a force to the ball, updating its acceleration, velocity,
        and position based on simple physics.
        """
        
        if disturbance:
            amp = 10.0
            freq = 1.0
            octaves = 4
            if isinstance(disturbance, (int, float)):
                amp = float(disturbance)
            elif isinstance(disturbance, dict):
                amp = float(disturbance.get('amplitude', amp))
                freq = float(disturbance.get('frequency', freq))
                octaves = int(disturbance.get('octaves', octaves))
            self._noise_time += config.TIME_STEP * freq
            # pnoise1 returns values in [-1, 1] (with octaves)
            disturbance = pnoise1(self._noise_time, octaves=octaves) * amp
        else:
            disturbance = 0.0
        net_force = force - config.GRAVITY * config.BALL_MASS + disturbance
        
        # Newton's second law: F = ma  =>  a = F/m
        acceleration = net_force / config.BALL_MASS
        
        # Update velocity based on acceleration over the time step
        self.velocity += acceleration * config.TIME_STEP
        
        # Update position based on the new velocity
        self.y += self.velocity * config.TIME_STEP

        # Handle collision with the ground
        if self.y < config.GROUND_HEIGHT + config.BALL_RADIUS:
            self.y = config.GROUND_HEIGHT + config.BALL_RADIUS
            self.velocity = 0

    def draw(self, screen):
        """Draws the ball on the pygame screen."""
        # Pygame's y-axis is inverted (0 is at the top), so we adjust the y-coordinate for drawing.
        draw_y = int(config.SCREEN_HEIGHT - self.y)
        draw_x = int(self.x)
        pygame.draw.circle(screen, config.BLUE, (draw_x, draw_y), config.BALL_RADIUS)
