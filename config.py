# --- Simulation Parameters ---
GRAVITY = 9.81
BALL_MASS = 1.0
TIME_STEP = 0.02  # 50 Hz simulation frequency
FORCE_MAGNITUDE = 20.0  # Max force applied by controller

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
PPO_TIMESTEPS = 20000000
MAX_EPISODE_STEPS = 2000

# --- Environment Settings ---
TARGET_HEIGHT = 300.0
RANDOM_REFERENCE = True #True/False
MOVING_REFERENCE = True #True/False
MOVING_REFERENCE_PERIODE = 0.05
MOVING_REFERENCE_AMPLITUDE = 0.2
