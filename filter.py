import numpy as np
import config
from scipy.optimize import minimize
from autograd import grad, jacobian


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
    def __init__(self, variances: np.ndarray):
        self.dt = config.TIME_STEP # discretization time
        self.nx = len(variances) # number of states
        self.Q = np.diag(variances) # process noise covariance matrix
    

    def f(self, x: np.ndarray, u) -> np.ndarray:
        """x^(k+1) = f(x^k, u^k)"""
        x.reshape(self.nx,1)
        A = np.array([
            [1, self.dt],
            [0, 1]
            ])
        B = np.array([
            [0, 0],
            [1/config.BALL_MASS, -config.GRAVITY]
        ])*self.dt
        u_mod = np.vstack([u,1]) # adding artificial input as gravity, in addition to real input
        x_next = np.stack(A@x + B@u_mod)
        return x_next
    

    def F(self, x: np.ndarray, u) -> np.ndarray:
        """Jacobian of dynamical function"""
        ff = lambda x, u: (self.f(x, u)).flatten()
        jac = jacobian(ff, 0)
        return jac(x, u).reshape(self.nx,self.nx)


    def predict_x(self, x_prev: gaussian, u) -> gaussian:
        F = self.F(x_prev.mean, u)
        x_pred_mean = self.f(x_prev.mean, u)
        x_pred_cov = F @ x_prev.cov @ F.T + self.Q
        x_pred = gaussian(x_pred_mean, x_pred_cov)
        return x_pred



class sensor_model:
    def __init__(self, variances):
        self.dt = config.TIME_STEP # discretization time
        self.nz = len(variances) # number of measurements
        self.R = np.diag(variances) # measurement noise covariance

    def h(self, x: np.ndarray) -> np.array:
        """z^k = h(x^k)"""
        x.reshape(len(x),1)
        return np.eye(self.nz, len(x))@x


    def H(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of measurement function"""
        return jacobian(self.h, 0)(x.flatten())


    def predict_z(self, x_pred: gaussian) -> gaussian:
        H = self.H(x_pred.mean)
        z_pred_mean = self.h(x_pred.mean)
        z_pred_cov = H @ x_pred.cov @ H.T + self.R
        z_pred = gaussian(z_pred_mean, z_pred_cov)
        return z_pred
    


# --- Extended Kalman Filter ---
#
# usage:
# initialization:
# ekf = EKF(
#     dynamic_model(),
#    sensor_model())
#
# running:
# est = run_ekf(ekf, z_meas, input)

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
        H = self.sens_mod.H(x_est_pred.mean)
        P = x_est_pred.cov
        S = z_est_pred.cov

        kalman_gain = P @ H.T @ np.linalg.inv(S)
        innovation = z - z_est_pred.mean

        state_upd_mean = x_est_pred.mean + kalman_gain @ innovation
        state_upd_cov = P - kalman_gain @ H @ P

        x_est_upd = gaussian(state_upd_mean, state_upd_cov)

        return x_est_upd



# --- Moving Horizon Estimator ---
#
# usage:
# initialization:
# mhe = MHE(
#     dynamic_model(),
#     sensor_model(),
#     mhe_horizon)
#
# running:
# est = run_mhe(mhe, z_meas, input)

class MHE:
    def __init__(self, dyn_mod: dynamic_model, sens_mod: sensor_model, M: int, dt=config.TIME_STEP):
        self.dt = dt # discretization time

        self.dyn_mod = dyn_mod
        self.sens_mod = sens_mod

        self.M = M  # maximum horizon length
        self.nx = self.dyn_mod.nx  # state dimension
        self.nz = self.sens_mod.nz  # measurement dimension

        # Weights used in cost function, sans arrival cost. Currently set as the inverse of process/measurement noise covariances
        self.W = np.linalg.inv(dyn_mod.Q)
        self.V = np.linalg.inv(sens_mod.R)

        # Buffers for states, inputs and measurements
        self.x_ests = np.empty((self.nx,0))
        self.z_buffer = np.empty((self.nz,0))
        self.u_buffer = []

        # Initialize prior. These values are immediately overwritten in main
        self.x_prior = np.zeros((self.nx, 1))
        self.P_prior = np.eye(self.nx)

        # Warm start (state trajectory guess)
        self.x_guess = None


    def set_arrival_cost(self, x_prior: np.ndarray, P_prior: np.ndarray):
        """Set the arrival cost prior state and covariance"""
        self.x_prior = x_prior
        self.P_prior = P_prior

    def add_measurement(self, z: np.ndarray, u):
        """Add new measurement and input"""
        self.z_buffer = np.append(self.z_buffer, z, axis=1)
        self.u_buffer.append(u)


    def cost_function(self, x_flat: np.ndarray, z_seq, u_seq, x_prior, P_prior_inv, N):
        """Compute the MHE cost for the current horizon"""
        X = x_flat.reshape(N, self.nx, 1)

        J = 0.0

        # Arrival cost term (first state)
        x0 = X[0].reshape(self.nx, 1)
        err_prior = x0 - x_prior
        J += float(err_prior.T @ P_prior_inv @ err_prior)

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
        H = self.sens_mod.H(self.x_prior)
        F = self.dyn_mod.F(self.x_prior, self.u_buffer[-1])
        P = self.P_prior
        Q = self.dyn_mod.Q
        R = self.sens_mod.R

        P_pred = F @ P @ F.T + Q
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        P_upd = (np.eye(self.nx) - K @ H) @ P_pred
        self.P_prior = P_upd

        if est_state:
            # Update state estimate using last measurement
            z_last = self.z_buffer.reshape(2,1)
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
        try:
            total_meas = self.z_buffer.shape[1]
        except IndexError as e:
            total_meas = 1

        # Determine active horizon length
        N = min(self.M, total_meas)

        if N == 1:
            self.kalman_update(est_state=True) # Horizon of 1 is just a Kalman update
            return self.x_prior
        
        # Extract last N measurements and inputs
        z_seq = self.z_buffer.T[-N:].reshape(N, self.nz, 1)
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
            P_prior_inv = np.linalg.inv(self.P_prior)
            res = minimize(
                self.cost_function,
                x0,
                args=(z_seq, u_seq, self.x_prior, P_prior_inv, N),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
        if optimizer == 1:
            raise NotImplementedError("Acados optimizer not implemented yet.")

        # Reshape optimized trajectory
        X_opt = res.x.reshape(N, self.nx, 1)
        self.x_guess = X_opt  # store for warm start next iteration

        # Update arrival cost (last state becomes new prior)
        self.x_prior = X_opt[-1]
        self.kalman_update(est_state=False) # update prior covariance only

        self.x_ests = np.append(self.x_ests, self.x_prior, axis=1)

        return self.x_prior
    


# -------------------------
# --- Utility functions ---
#--------------------------

# --- Adding noise to measurements ---
def add_noise(pos, vel):
    z_pos = pos + np.random.normal(0, config.STD_POS)
    z_vel = vel + np.random.normal(0, config.STD_VEL)
    z_meas = np.vstack([z_pos, z_vel])

    return z_meas

def init_estimator(estimator: int):
    """
    2: ekf\n
    3: mhe
    """
    if estimator == 2:
        return EKF(
            dynamic_model(np.array([config.EKF_VAR_PROC_POS, config.EKF_VAR_PROC_VEL])),
            sensor_model(([config.EKF_VAR_MEAS_POS, config.EKF_VAR_MEAS_VEL]))
            )
    if estimator == 3:
        return MHE(
            dynamic_model(np.array([config.EKF_VAR_PROC_POS, config.EKF_VAR_PROC_VEL])),
            sensor_model(([config.EKF_VAR_MEAS_POS, config.EKF_VAR_MEAS_VEL])),
            config.MHE_HORIZON
            )


def run_ekf(ekf: EKF, z: np.ndarray, odometry: list):
    u = odometry[-1] if len(odometry) > 0 else 0
    if len(odometry) == 0:
        # Initialize EKF with first measurement
        init_mean = np.vstack([z[0], z[1]])
        init_cov = np.diag([1, 0.5])
        x_est = gaussian(init_mean, init_cov)
        x_est_pred, z_est_pred = ekf.predict(x_est, u)
    else:
        # EKF predict and update steps
        x_est_pred, z_est_pred = ekf.predict(ekf.state_ests[-1], u)
        x_est = ekf.update(x_est_pred, z_est_pred, z)
    est_pos, est_vel = x_est.mean
    ekf.meas_ests.append(z_est_pred)
    ekf.state_ests.append(x_est)

    return est_pos, est_vel


def run_mhe(mhe: MHE, z:np.ndarray, odometry: list):
    u = odometry[-1] if len(odometry) > 0 else 0
    mhe.add_measurement(z, u)
    if len(odometry) == 0:
        # Initialize MHE with first measurement
        init_mean = np.vstack([z[0], z[1]])
        init_cov = np.diag([1*6, 0.5*6])
        mhe.set_arrival_cost(init_mean, init_cov)
    x_est = mhe.solve(0)

    return x_est

filter_dict = {
1: "GT",
2: "EKF",
3: "MHE"
}

controller_dict = {
1: "P",
2: "PPO",
3: "MPC",
4: "SMPC",
5: "TMPC"
}