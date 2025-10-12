import numpy as np
import config



class gaussian:
    def __init__(self, mean: np.array, cov: np.array):
        self.mean = mean
        self.cov = cov


    def mahalanobis_distance(self, x: np.ndarray) -> float:
        # --- Normalized distance from mean ---

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
    

    def f(self, x: np.array, u) -> np.array: # x^(k+1) = f(x^k, u^k)
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
    def __init__(self, dynamic_model, sensor_model, dt=config.TIME_STEP):
        self.dt = dt # discretization time
        self.dynamic_model = dynamic_model
        self.sensor_model = sensor_model


    def predict(self, x_est_prev: gaussian, u) -> tuple[gaussian, gaussian]:
        # ---Perform one EKF prediction step---
        x_est_pred = self.dynamic_model.predict_x(x_est_prev, u)
        z_est_pred = self.sensor_model.predict_z(x_est_pred)
        return x_est_pred, z_est_pred
    

    def update(self, x_est_pred: gaussian, z_est_pred: gaussian, z) -> gaussian:
        # ---Perform one EKF update step---
        H = self.sensor_model.H
        P = x_est_pred.cov
        S = z_est_pred.cov

        kalman_gain = P @ H.T @ np.linalg.inv(S)
        innovation = z - z_est_pred.mean

        state_upd_mean = x_est_pred.mean + kalman_gain @ innovation
        state_upd_cov = P - kalman_gain @ H @ P

        x_est_upd = gaussian(state_upd_mean, state_upd_cov)

        return x_est_upd


class MHE:
    def __init__(self, N: int):
        self.N = N # horizon length
        self.n = 2 # state dimension
        self.m = 2 # measurement dimension

        self.Q = np.diag([0.1, 0.1]) # process noise covariance
        self.R = np.diag([0.5, 0.5]) # measurement noise covariance