from casadi import *
import numpy as np
import config

class MPCControllerTube:
    """
    Tube MPC: nominal open-loop MPC + linear feedback (LQR) for tube.
    Usage: create instance and call get_action(current_height, current_velocity).
    Returns: applied_force, predicted_nominal_X, predicted_nominal_U
    """
    def __init__(self, N=config.TUBE_MPC_HORIZON, dt=config.TIME_STEP):
        self.N = N
        self.dt = dt
        self.opti = Opti()

        # Nominal trajectory variables
        self.Xn = self.opti.variable(2, N + 1)  # nominal states
        self.Un = self.opti.variable(1, N)      # nominal controls

        # Parameters
        self.X0 = self.opti.parameter(2)        # real current state (used as init for nominal)
        self.target_height = self.opti.parameter()

        # Linearized discrete-time model (same as in mpc_controller)
        A = np.array([[1.0, dt],
                      [0.0, 1.0]])
        B = np.array([[0.0],
                      [dt]])

        # Dynamics constraints for nominal trajectory
        for k in range(N):
            h_next = self.Xn[0, k] + self.Xn[1, k] * dt
            v_next = self.Xn[1, k] + (self.Un[0, k]/config.BALL_MASS - config.GRAVITY) * dt
            self.opti.subject_to(self.Xn[0, k + 1] == h_next)
            self.opti.subject_to(self.Xn[1, k + 1] == v_next)

        # Initial condition for nominal trajectory set to parameter
        self.opti.subject_to(self.Xn[:, 0] == self.X0)

        # Control bounds for nominal
        self.lbu = -config.FORCE_MAGNITUDE
        self.ubu = config.FORCE_MAGNITUDE
        # tighten nominal bounds slightly for robustness margin (tube)
        self.u_tightening = config.TUBE_MPC_TIGHTING_U  # tuning parameter: tighten control by this much
        self.opti.subject_to(self.opti.bounded(self.lbu + self.u_tightening, self.Un, self.ubu - self.u_tightening))

        # Optional delta-u constraint on nominal
        self.delta_u_max = config.TUBE_MPC_DELTA_U_MAX
        for k in range(1, N):
            delta_u = self.Un[0, k] - self.Un[0, k-1]
            self.opti.subject_to(self.opti.bounded(-self.delta_u_max, delta_u, self.delta_u_max))

        # State constraints (tighten by margin)
        x_tightening = config.TUBE_MPC_TIGHTING_X  # height tightening
        self.opti.subject_to(self.Xn[0, :] >= 0 + x_tightening)

        # Cost (nominal)
        qh = config.TUBE_MPC_QH
        qv = config.TUBE_MPC_QV
        r = config.TUBE_MPC_R
        Q = np.diag([qh, qv])
        R = np.diag([r])
        cost = 0
        for k in range(N):
            state_err = self.Xn[:, k] - vertcat(self.target_height, 0)
            cost += mtimes([state_err.T, Q, state_err]) + mtimes([self.Un[:, k].T, R, self.Un[:, k]])
        terminal_err = self.Xn[:, N] - vertcat(self.target_height, 0)
        Q_terminal = config.TUBE_MPC_TERMINAL * Q
        cost += mtimes([terminal_err.T, Q_terminal, terminal_err])
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

        # Initial guess
        self.opti.set_initial(self.Xn, np.zeros((2, N + 1)))
        self.opti.set_initial(self.Un, np.full((1, N), config.GRAVITY))

        # Precompute LQR feedback K for the linearized model (infinite horizon approx)
        self.K = self._compute_lqr_gain(A, B, Q * config.TUBE_MPC_LQR_SCALE, R)

    def _compute_lqr_gain(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, max_iters=500, eps=1e-8):
        """
        Solve discrete-time algebraic Riccati equation iteratively and return K = (R+B'PB)^-1 B' P A
        """
        P = Q.copy()
        for _ in range(max_iters):
            P_next = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
            if np.max(np.abs(P_next - P)) < eps:
                P = P_next
                break
            P = P_next
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        # return shape (1,2)
        return K

    def get_action(self, current_height, current_velocity, target_height, return_trajectory=True):
        """Solve nominal MPC and apply tube control u = u_nom + K (x - x_nom)."""
        # set parameters
        self.opti.set_value(self.X0, [current_height, current_velocity])
        self.opti.set_value(self.target_height, target_height)

        try:
            sol = self.opti.solve()
            Un_opt = sol.value(self.Un)  # (1, N) but may be flattened to (N,)
            Xn_opt = sol.value(self.Xn)  # (2, N+1)
            u_nom0 = float(Un_opt.flat[0]) 
            x_nom0 = Xn_opt[:, 0].reshape(-1)
            # compute feedback correction
            x_curr = np.array([current_height, current_velocity]).reshape(-1)
            delta = x_curr - x_nom0
            u_corr = float((self.K @ delta).reshape(()))
            u_applied = u_nom0 + u_corr
            # saturate
            u_applied = float(np.clip(u_applied, self.lbu, self.ubu))
            # store last solution
            self.last_solution = {'Xn': Xn_opt, 'Un': Un_opt}
            if return_trajectory:
                return u_applied, Xn_opt, Un_opt
            else:
                return u_applied, None, None
        except RuntimeError as e:
            # fallback to gravity compensation
            print(f"Tube MPC solver failed: {e}")
            return config.GRAVITY, None, None

    def sizes(self):
        """Return tuning/limits for diagnostics similar to other controllers."""
        return config.STD_MPC_QH, config.STD_MPC_QV, self.lbu, self.ubu, config.STD_MPC_R, self.delta_u_max