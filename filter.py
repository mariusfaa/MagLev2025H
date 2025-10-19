import numpy as np
import config
from scipy.optimize import minimize


class gaussian:
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = mean
        self.cov = cov


    def mahalanobis_distance(self, x: np.ndarray) -> float:
        """Normalized distance from mean"""

        err = x.reshape(-1, 1) - self.mean.reshape(-1, 1)
        mahalanobis_distance = float(err.T @ np.linalg.solve(self.cov, err))
        return mahalanobis_distance


class dynamic_model:
    def __init__(self, dt=config.TIME_STEP, var_pos=0.1, var_vel=0.1):
        self.dt = dt # discretization time
        self.g = config.GRAVITY # gravity constant
        self.m = config.BALL_MASS # mass of the ball
        self.var_pos = var_pos # process noise variance for position
        self.var_vel = var_vel # process noise variance for velocity

        self.F = np.array([
            [1, self.dt],
            [0, 1]
        ]) # Jacobian of model function

        self.Q = np.diag([self.var_pos, self.var_vel]) # process noise covariance
    

    def f(self, x: np.ndarray, u) -> np.ndarray: # x^(k+1) = f(x^k, u^k)
        A = self.F
        B = np.array([
            [0, 0],
            [1/self.g, -self.dt*self.g]
        ])
        u_mod = np.vstack([u,1]) # adding artificial input as gravity, in addition to real input
        x_next = A@x + B@u_mod
        return x_next


    def predict_x(self, x_prev: gaussian, u) -> gaussian:
        x_pred_mean = self.f(x_prev.mean, u)
        x_pred_cov = self.F @ x_prev.cov @ self.F.T + self.Q
        x_pred = gaussian(x_pred_mean, x_pred_cov)
        return x_pred



class sensor_model:
    def __init__(self, dt=config.TIME_STEP, var_meas_pos=0.5, var_meas_vel=0.5):
        self.dt = dt # discretization time
        self.var_meas_pos = var_meas_pos # measurement noise variance for position
        self.var_meas_vel = var_meas_vel # measurement noise variance for velocity
        
        self.H = np.eye(2,2) # Jacobian of measurement function
        
        self.R = np.diag([self.var_meas_pos, self.var_meas_vel]) # measurement noise covariance


    def h(self, x: np.array) -> np.array: # z^k = h(x^k)
        return self.H@x
    

    def predict_z(self, x_pred: gaussian) -> gaussian:
        z_pred_mean = self.h(x_pred.mean)
        z_pred_cov = self.H @ x_pred.cov @ self.H.T + self.R
        z_pred = gaussian(z_pred_mean, z_pred_cov)
        return z_pred
    


class EKF:
    def __init__(self, dyn_mod: dynamic_model, sens_mod: sensor_model, dt=config.TIME_STEP):
        self.dt = dt # discretization time
        self.dyn_mod = dyn_mod
        self.sens_mod = sens_mod

        self.state_ests = []  # buffer for estimated states
        self.meas_ests = []   # buffer for estimated measurements


    def predict(self, x_est_prev: gaussian, u) -> tuple[gaussian, gaussian]:
        """Perform one EKF prediction step"""
        x_est_pred = self.dyn_mod.predict_x(x_est_prev, u)
        z_est_pred = self.sens_mod.predict_z(x_est_pred)
        return x_est_pred, z_est_pred
    

    def update(self, x_est_pred: gaussian, z_est_pred: gaussian, z) -> gaussian:
        """Perform one EKF update step"""
        H = self.sens_mod.H
        P = x_est_pred.cov
        S = z_est_pred.cov

        kalman_gain = P @ H.T @ np.linalg.inv(S)
        innovation = z - z_est_pred.mean

        state_upd_mean = x_est_pred.mean + kalman_gain @ innovation
        state_upd_cov = P - kalman_gain @ H @ P

        x_est_upd = gaussian(state_upd_mean, state_upd_cov)

        return x_est_upd


class MHE:
    def __init__(self, dyn_mod: dynamic_model, sens_mod: sensor_model, M: int, dt=config.TIME_STEP):
        self.dt = dt # discretization time

        self.dyn_mod = dyn_mod
        self.sens_mod = sens_mod

        self.M = M  # maximum horizon length
        self.n = self.dyn_mod.F.ndim  # state dimension
        self.m = self.sens_mod.H.ndim  # measurement dimension

        # Weights used in cost function, sans arrival cost. Currently set as process/measurement noise covariances
        self.W = np.linalg.inv(dyn_mod.Q)
        self.V = np.linalg.inv(sens_mod.R)

        # Buffers for states, inputs and measurements
        self.x_ests = np.empty((2,0))
        self.u_buffer = []
        self.z_buffer = []

        # Initialize prior. These values are immediately overwritten in main
        self.x_prior = np.zeros((self.n, 1))
        self.P_prior = np.eye(self.n)

        # Warm start (state trajectory guess)
        self.x_guess = None


    def set_arrival_cost(self, x_prior: np.ndarray, P_prior: np.ndarray):
        """Set the arrival cost prior state and covariance"""
        self.x_prior = x_prior
        self.P_prior = P_prior

    def add_measurement(self, z: np.ndarray, u: np.ndarray):
        """Add new measurement and input"""
        self.z_buffer.append(z)
        self.u_buffer.append(u)


    def cost_function(self, x_flat: np.ndarray, z_seq, u_seq, x_prior, P_prior):
        """Compute the MHE cost for the current horizon"""
        N = len(z_seq)
        X = x_flat.reshape(N, self.n, 1)

        P_inv = np.linalg.inv(P_prior)

        J = 0.0

        # Arrival cost term (first state)
        x0 = X[0]
        err_prior = x0 - x_prior
        J += float(err_prior.T @ P_inv @ err_prior)

        # Process and measurement terms
        for i in range(N):
            z_i = z_seq[i]
            z_pred = self.sens_mod.h(X[i])
            err_z = z_i - z_pred
            J += float(err_z.T @ self.V @ err_z)

            if i < N - 1:
                u_i = u_seq[i]
                x_pred = self.dyn_mod.f(X[i], u_i)
                proc_noise = X[i + 1] - x_pred
                J += float(proc_noise.T @ self.W @ proc_noise)

        return J
    
    
    def kalman_update(self, est_state: bool):
        """Kalman based update to the prior covariance, and state if needed"""
        H = self.sens_mod.H
        F = self.dyn_mod.F
        P = self.P_prior
        Q = self.dyn_mod.Q
        R = self.sens_mod.R

        P_pred = F @ P @ F.T + Q
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        P_upd = (np.eye(self.n) - K @ H) @ P_pred
        self.P_prior = P_upd

        if est_state:
            # Update state estimate using last measurement
            z_last = self.z_buffer[-1]
            z_pred = self.sens_mod.h(self.x_prior)
            innovation = z_last - z_pred
            self.x_prior = self.x_prior + K @ innovation


    def solve(self, optimizer: int):
        """
        Solve the nonlinear MHE optimization problem\n
        Optimizer:\n
        0 for scipy minimize\n
        1 for Acados (not implemented)
        """
        total_meas = len(self.z_buffer)

        # Determine active horizon length
        N = min(self.M, total_meas)

        if (total_meas < 2) or N == 1:
            self.kalman_update(est_state=True) # Either case corresponds to a horizon of 1, which is just a Kalman update
            return self.x_prior

        # Extract last N measurements and inputs
        z_seq = self.z_buffer[-N:]
        u_seq = self.u_buffer[-N:]

        # Initialize trajectory guess
        if self.x_guess is None or len(self.x_guess) != N:
            self.x_guess = np.tile(self.x_prior, (N, 1))
        else:
            # Adjust guess length as horizon grows or slides
            if len(self.x_guess) > N:
                self.x_guess = self.x_guess[-N:]
            elif len(self.x_guess) < N:
                add = np.tile(self.x_guess[-1], (N - len(self.x_guess), 1))
                self.x_guess = np.vstack([self.x_guess, add])

        x0 = self.x_guess.flatten()

        # ---Optimize cost function---
        
        # Using scipy minimize
        if optimizer == 0:
            res = minimize(
                self.cost_function,
                x0,
                args=(z_seq, u_seq, self.x_prior, self.P_prior),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
        if optimizer == 1:
            raise NotImplementedError("Acados optimizer not implemented yet.")

        # Reshape optimized trajectory
        X_opt = res.x.reshape(N, self.n, 1)
        self.x_guess = X_opt  # store for warm start next iteration

        # Update arrival cost (last state becomes new prior)
        self.x_prior = X_opt[-1]
        self.kalman_update(est_state=False) # update prior covariance only

        self.x_ests = np.append(self.x_ests, self.x_prior, axis=1)

        return self.x_prior