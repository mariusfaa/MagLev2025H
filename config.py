import math

# --- Simulation Parameters ---
GRAVITY = 9.81
BALL_MASS = 1.0
TIME_STEP = 0.02  # 50 Hz simulation frequency
FORCE_MAGNITUDE = 60.0  # Max force applied by controller
# TARGET_HEIGHT = 300.0

TARGET_MEAN = 300.0
TARGET_HEIGHT = TARGET_MEAN
TARGET_AMPLITUDE = 50.0
TARGET_PERIOD = 5.0

def target_height(t: float) -> float:
    """Returns a time-varying target height based on a sine wave."""
    return TARGET_MEAN + TARGET_AMPLITUDE * math.sin(2 * math.pi * t / TARGET_PERIOD)

# --- Pygame Visualization Parameters ---
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
BALL_RADIUS = 20
GROUND_HEIGHT = 50

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# --- RL Training Parameters ---
PPO_TIMESTEPS = 2000000
MAX_EPISODE_STEPS = 1500
