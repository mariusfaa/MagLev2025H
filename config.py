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

# --- Measurement Noise ---
STD_POS = 6
STD_VEL = 3

# --- EKF Parameters ---
EKF_VAR_PROC_POS = 2
EKF_VAR_PROC_VEL = 2
EKF_VAR_MEAS_POS = 36
EKF_VAR_MEAS_VEL = 9

# --- MHE Parameters ---
MHE_HORIZON = 2
MHE_VAR_PROC_POS = 2
MHE_VAR_PROC_VEL = 2
MHE_VAR_MEAS_POS = 36
MHE_VAR_MEAS_VEL = 9

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
