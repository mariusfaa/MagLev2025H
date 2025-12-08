from casadi import *
import numpy as np
import config

class MPCController:
    """A Model Predictive Controller (MPC) for the ball environment."""
    def __init__(self, N=config.STD_MPC_HORIZON, dt=config.TIME_STEP):
        """Initializes the MPC controller with a prediction horizon (N) and time step (dt)."""
        self.N = N
        self.dt = dt
        self.opti = Opti()

        # Define state and control variables
        self.X = self.opti.variable(2, N + 1)  # States: [height, velocity]
        self.U = self.opti.variable(1, N)      # Control: [force]

        # Parameters (initial state and target height)
        self.X0 = self.opti.parameter(2)       # Initial state
        self.target_height = self.opti.parameter()  # Target height

        # Dynamics constraints
        for k in range(N):
            h_next = self.X[0, k] + self.X[1, k] * dt
            v_next = self.X[1, k] + (self.U[0, k]/config.BALL_MASS - config.GRAVITY) * dt
            self.opti.subject_to(self.X[0, k + 1] == h_next)
            self.opti.subject_to(self.X[1, k + 1] == v_next)

        # Initial condition constraint
        self.opti.subject_to(self.X[:, 0] == self.X0)

        # Control constraints
        self.lbu = config.FORCE_MAGNITUDE * -1
        self.ubu = config.FORCE_MAGNITUDE
        self.opti.subject_to(self.opti.bounded(self.lbu, self.U, self.ubu))
        
        self.delta_u_max = config.STD_MPC_DELTA_U_MAX  # Set your desired max change

        for k in range(1, N):
            delta_u = self.U[0, k] - self.U[0, k-1]
            self.opti.subject_to(self.opti.bounded(-self.delta_u_max, delta_u, self.delta_u_max))

        # State constraints (optional - uncomment if needed)
        self.opti.subject_to(self.X[0, :] >= 0)  # Height must be non-negative

        # Objective function: minimize the distance to the target height and control effort
        self.qh = config.STD_MPC_QH
        self.qv = config.STD_MPC_QV
        self.r = config.STD_MPC_R
        Q = np.diag([self.qh, self.qv])  # State cost weights
        R = np.diag([self.r])            # Control cost weight

        cost = 0
        for k in range(N):
            state_error = self.X[:, k] - vertcat(self.target_height, 0)
            cost += mtimes([state_error.T, Q, state_error]) + mtimes([self.U[:, k].T, R, self.U[:, k]])
        
        # Terminal cost (higher weight for final state)
        terminal_error = self.X[:, N] - vertcat(self.target_height, 0)
        Q_terminal = config.STD_MPC_TERMINAL * Q
        cost += mtimes([terminal_error.T, Q_terminal, terminal_error])

        self.opti.minimize(cost)

        # Solver options
        ipopt_opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            # "ipopt.max_iter": 300,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4
        }
        # self.opti.solver('qpOASES', qpOASES_opts)
        self.opti.solver('ipopt', ipopt_opts)

        # Initialize with reasonable guess
        self.opti.set_initial(self.X, np.zeros((2, N + 1)))
        self.opti.set_initial(self.U, np.full((1, N), config.GRAVITY))  # Start with gravity compensation
        
        # Store last solution for warm starting
        self.last_solution = None

    def get_action(self, current_height, current_velocity, target_height, return_trajectory=True):
        """Computes the optimal control action given the current state."""
        self.opti.set_value(self.X0, [current_height, current_velocity])
        self.opti.set_value(self.target_height, target_height)

        try:
            sol = self.opti.solve()
            optimal_force = sol.value(self.U[0, 0])
            # Save solution for next iteration (warm start / diagnostics)
            self.last_solution = {
                'X': sol.value(self.X),
                'U': sol.value(self.U)
            }
            predicted_X = sol.value(self.X)  # shape (2, N+1)
            predicted_U = sol.value(self.U)  # shape (1, N)
            
        except RuntimeError as e:
            # If the solver fails, return gravity compensation
            print(f"MPC solver failed: {e}")
            optimal_force = config.GRAVITY  # Hover in place
            # No predicted trajectory available on failure
            self.last_solution = None
            return optimal_force, None, None

        if return_trajectory:
            return optimal_force, predicted_X, predicted_U
        else:
            return optimal_force, None, None
    
    def sizes(self):
        """Returns the sizes of the state and action spaces."""
        return self.qh, self.qv, self.lbu, self.ubu, self.r, self.delta_u_max