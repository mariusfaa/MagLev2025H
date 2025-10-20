# --- Simulation Parameters ---
GRAVITY = 9.81
BALL_MASS = 1.0
TIME_STEP = 0.02  # 50 Hz simulation frequency
FORCE_MAGNITUDE = 20.0  # Max force applied by controller
TARGET_HEIGHT = 300.0

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

# --- Standard MPC Parameters ---
STD_MPC_HORIZON = 5
STD_MPC_QH = 100
STD_MPC_QV = 10
STD_MPC_R = 3
STD_MPC_TERMINAL = 5
STD_MPC_DELTA_U_MAX = 5

# --- Stochastic MPC Parameters ---
STOCHASTIC_MPC_HORIZON = 5
STOCHASTIC_MPC_SAMPLES = 15
STOCHASTIC_MPC_QH = 100
STOCHASTIC_MPC_QV = 10
STOCHASTIC_MPC_R = 1
STOCHASTIC_MPC_TERMINAL = 5
STOCHASTIC_MPC_DELTA_U_MAX = 5