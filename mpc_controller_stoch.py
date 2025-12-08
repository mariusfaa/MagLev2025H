from casadi import *
import numpy as np
import config

class MPCControllerStochastic:
    """A Model Predictive Controller (MPC) for the ball environment."""
    def __init__(self, N=config.STOCHASTIC_MPC_HORIZON, dt=config.TIME_STEP, num_samples=config.STOCHASTIC_MPC_SAMPLES, noise_std=config.STOCHASTIC_MPC_NOISE_STD):
        """Initializes the Stochastic MPC controller with Monte Carlo simulation."""
        self.N = N
        self.dt = dt
        self.num_samples = num_samples
        self.noise_std = noise_std  # Standard deviation for process noise
        self.opti = Opti()

        # Control variables (shared across all samples)
        self.U = self.opti.variable(1, N)      # Control: [force]

        # Parameters (initial state and target height)
        self.X0 = self.opti.parameter(2)       # Initial state
        self.target_height = self.opti.parameter()  # Target height

        # Create list of noise parameters for each sample
        self.Noise_params = [self.opti.parameter(2, N) for _ in range(num_samples)]
        
        # For each sample, define its own state trajectory
        self.X_samples = [self.opti.variable(2, N + 1) for _ in range(num_samples)]

        # Initial condition constraints for all samples
        for i in range(num_samples):
            self.opti.subject_to(self.X_samples[i][:, 0] == self.X0)

        # Control constraints (shared)
        self.lbu = config.FORCE_MAGNITUDE * -1
        self.ubu = config.FORCE_MAGNITUDE
        self.opti.subject_to(self.opti.bounded(self.lbu, self.U, self.ubu))

        self.delta_u_max = config.STOCHASTIC_MPC_DELTA_U_MAX  # Set your desired max change
        for k in range(1, N):
            delta_u = self.U[0, k] - self.U[0, k-1]
            self.opti.subject_to(self.opti.bounded(-self.delta_u_max, delta_u, self.delta_u_max))

        # State constraints (optional)
        for i in range(num_samples):
            self.opti.subject_to(self.X_samples[i][0, :] >= 0)  # Height must be non-negative

        # Objective function: expected cost over all samples
        self.qx = config.STOCHASTIC_MPC_QH
        self.qu = config.STOCHASTIC_MPC_QV
        self.r = config.STOCHASTIC_MPC_R
        Q = np.diag([self.qx, self.qu])  # State cost weights
        R = np.diag([self.r])            # Control cost weight

        # For reproducibility, use fixed noise samples
        # np.random.seed(42)
        # self.noise_samples = [np.random.normal(0, self.noise_std, (2, N)) for _ in range(num_samples)]

        cost = 0
        for i in range(num_samples):
            X = self.X_samples[i]
            noise = self.Noise_params[i]
            # Dynamics constraints for this sample
            for k in range(N):
                # Additive process noise to both height and velocity
                h_next = X[0, k] + X[1, k] * dt + noise[0, k]
                v_next = X[1, k] + (self.U[0, k]/config.BALL_MASS - config.GRAVITY) * dt + noise[1, k]
                self.opti.subject_to(X[0, k + 1] == h_next)
                self.opti.subject_to(X[1, k + 1] == v_next)
            # Cost for this sample
            for k in range(N):
                state_error = X[:, k] - vertcat(self.target_height, 0)
                cost += mtimes([state_error.T, Q, state_error]) + mtimes([self.U[:, k].T, R, self.U[:, k]])
            # Terminal cost
            terminal_error = X[:, N] - vertcat(self.target_height, 0)
            Q_terminal = config.STOCHASTIC_MPC_TERMINAL * Q
            cost += mtimes([terminal_error.T, Q_terminal, terminal_error])
        # Average cost over all samples
        cost = cost / num_samples
        self.opti.minimize(cost)

        # Solver options
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 300,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4
        }
        self.opti.solver('ipopt', opts)

        # Initialize with reasonable guess
        for i in range(num_samples):
            self.opti.set_initial(self.X_samples[i], np.zeros((2, N + 1)))
        self.opti.set_initial(self.U, np.full((1, N), config.GRAVITY))  # Start with gravity compensation
        # Store last solution for warm starting

    def get_action(self, current_height, current_velocity, target_height, return_trajectory=True):
        """Computes the optimal control action given the current state, using stochastic MPC."""
        self.opti.set_value(self.X0, [current_height, current_velocity])
        self.opti.set_value(self.target_height, target_height)
        
        for i in range(self.num_samples):
            fresh_noise = np.random.normal(0, self.noise_std, (2, self.N))
            self.opti.set_value(self.Noise_params[i], fresh_noise)

        try:
            sol = self.opti.solve()
            optimal_force = sol.value(self.U[0, 0])
            # Save solution for next iteration (warm start / diagnostics)
            self.last_solution = {
                'X_samples': [sol.value(X) for X in self.X_samples],
                'U': sol.value(self.U)
            }
            predicted_X_samples = [sol.value(X) for X in self.X_samples]  # list of (2, N+1)
            predicted_X_mean = np.mean(np.stack(predicted_X_samples, axis=0), axis=0)  # shape (2, N+1)
            predicted_U = sol.value(self.U)  # shape (1, N)
        except RuntimeError as e:
            print(f"Stochastic MPC solver failed: {e}")
            optimal_force = config.GRAVITY
            self.last_solution = None
            return optimal_force, None, None
        if return_trajectory:
            return optimal_force, predicted_X_mean, predicted_U
        else:
            return optimal_force, None, None
    
    def sizes(self):
        """Returns the sizes of the state and action spaces."""
        return self.qx, self.qu, self.lbu, self.ubu, self.r, self.delta_u_max