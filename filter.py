import numpy as np
import config
from scipy.optimize import minimize
from scipy.linalg import block_diag, cholesky, solve_triangular
from autograd import grad, jacobian
import casadi as ca
from acados_template import AcadosModel, AcadosOcpSolver, AcadosOcp, AcadosOcpOptions


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
    def __init__(self, Q: np.ndarray):
        self.dt = config.TIME_STEP  # discretization time
        self.nx = Q.shape[0]  # number of states
        self.Q = Q  # process noise covariance matrix

    def f(self, x: np.ndarray, u) -> np.ndarray:
        """x^(k+1) = f(x^k, u^k)"""
        x.reshape(self.nx, 1)
        A = np.array([
            [1, self.dt],
            [0, 1]
        ])
        B = np.array([
            [0, 0],
            [1/config.BALL_MASS, -config.GRAVITY]
        ])*self.dt
        u_mod = np.vstack([u, 1])  # adding artificial gravity as artificial input
        x_next = np.stack(A@x + B@u_mod)
        return x_next
    
    def acados_model(self) -> AcadosModel:

        model = AcadosModel()

        model.name = 'ball'

        # states
        x1 = ca.SX.sym('x1')
        x2 = ca.SX.sym('x2')
        x = ca.vertcat(x1, x2)

        # control input
        u = ca.SX.sym('u', 1)
        u_mod = ca.vertcat(u, 1)  # adding gravity as artificial input

        # state noise
        w_x1 = ca.SX.sym('w_x1')
        w_x2 = ca.SX.sym('w_x2')
        w = ca.vertcat(w_x1, w_x2)

        # xdot for implicit solvers
        x1_dot = ca.SX.sym('x1_dot')
        x2_dot = ca.SX.sym('x2_dot')
        x_dot = ca.vertcat(x1_dot, x2_dot)

        # continuous-time dynamics
        Ac = ca.DM([
            [0, 1],
            [0, 0]
        ])
        Bc = ca.DM([
            [0, 0],
            [1/config.BALL_MASS, -config.GRAVITY]
        ])
        f_expl = ca.mtimes(Ac, x) + ca.mtimes(Bc, u_mod)

        # discrete-time dynamics
        A = ca.DM([
            [1, self.dt],
            [0, 1]
        ])
        B = ca.DM([
            [0, 0],
            [1/config.BALL_MASS, -config.GRAVITY]
        ])*self.dt
        f_disc = ca.mtimes(A, x) + ca.mtimes(B, u_mod)

        # adding additive state noise
        #f_expl += w
        f_disc += w

        model.disc_dyn_expr = f_disc
        #model.f_expl_expr = f_expl
        #model.f_impl_expr = x_dot - f_expl
        model.x = x
        model.xdot = x_dot
        model.u = w
        model.p = u
        
        return model

    
    
    def F(self, x: np.ndarray, u) -> np.ndarray:
        """Jacobian of dynamical function"""
        jac = jacobian(lambda x, u: (self.f(x, u)).flatten(), 0)
        return jac(x, u).reshape(self.nx, self.nx)

    def predict_x(self, x_prev: gaussian, u) -> gaussian:
        F = self.F(x_prev.mean, u)
        x_pred_mean = self.f(x_prev.mean, u)
        x_pred_cov = F @ x_prev.cov @ F.T + self.Q
        x_pred = gaussian(x_pred_mean, x_pred_cov)
        return x_pred


class sensor_model:
    def __init__(self, R: np.ndarray):
        self.dt = config.TIME_STEP  # discretization time
        self.nz = R.shape[0]  # number of measurements
        self.R = R  # measurement noise covariance

    def h(self, x: np.ndarray) -> np.array:
        """z^k = h(x^k)"""
        x.reshape(len(x), 1)
        return np.eye(self.nz, len(x))@x
    
    def acados_model(self) -> AcadosModel:

        model = AcadosModel()

        model.z

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

class EKF:
    def __init__(self, dyn_mod: dynamic_model, sens_mod: sensor_model, dt=config.TIME_STEP):
        self.dt = dt  # discretization time
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

class MHE_acados:
    def __init__(
            self, dyn_mod: dynamic_model,
            sens_mod: sensor_model,
            M: int,
            Q0: np.ndarray,
            dt=config.TIME_STEP):

        self.dt = dt       # discretization time
        self.M = M         # maximum horizon length
        self.P_prior = Q0  # prior covariance

        # extract model info
        self.dyn_mod = dyn_mod
        self.sens_mod = sens_mod
        self.model = dyn_mod.acados_model()

        self.nx = self.model.x.rows()
        self.nu = self.model.p.rows()
        self.nz = self.model.x.rows() # TODO change if measurement model changes
        self.nparam = self.model.p.rows()

        self.ny_0 = self.nz + 2*self.nx   # h(x), w and arrival cost
        self.ny = self.nz + self.nx     # h(x), w

        # Buffers for states, measurements and inputs
        self.x_ests = np.empty((self.nx, 0))
        self.w_buffer = np.empty((self.nx, 0)) # maybe useful
        self.z_buffer = np.empty((self.nx, 0))
        self.u_buffer = [0]

        # initial states
        self.x0_bar =  np.zeros((self.nx,))
        self.w =  np.zeros((self.nx,))

        def _build_solver(self):
            W = np.linalg.inv(self.dyn_mod.Q)
            V = np.linalg.inv(self.sens_mod.R)

            ocp = AcadosOcp()
            ocp.model = self.model

            ocp.dims.N = self.M
            ocp.solver_options.tf = self.M * self.dt

            # cost type nonlinear least squares
            ocp.cost.cost_type_0 = 'NONLINEAR_LS'
            ocp.cost.cost_type = 'NONLINEAR_LS'
            ocp.cost.cost_type_e = 'LINEAR_LS'

            ocp.model.cost_y_expr_0 = ca.vertcat(self.model.x, self.model.u, self.model.x) # TODO  change to z
            ocp.model.cost_y_expr = ca.vertcat(self.model.x, self.model.u) # TODO change to z
            ocp.cost.W_0 = block_diag(V, W, Q0)
            ocp.cost.W = block_diag(V, W)
            ocp.cost.yref_0 = np.zeros((self.ny_0,))
            ocp.cost.yref = np.zeros((self.ny,))
            ocp.cost.yref_e = np.zeros(0)
            ocp.parameter_values = np.zeros((self.nparam, ))

            ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
            ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
            ocp.solver_options.integrator_type = 'DISCRETE'
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
            ocp.solver_options.nlp_solver_max_iter = 50
            ocp.solver_options.cost_scaling = np.ones((self.M + 1,))

            solver = AcadosOcpSolver(ocp, json_file='acados_mhe.json')
            return solver
        
        self.solver = _build_solver(self)

    def kalman_update(self, x: np.ndarray, u) -> np.ndarray:
        """Kalman based update to the prior covariance"""
        H = self.sens_mod.H(x)
        F = self.dyn_mod.F(x, u)
        P = self.P_prior
        Q = self.dyn_mod.Q
        R = self.sens_mod.R

        P_pred = F @ P @ F.T + Q
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        P_upd = (np.eye(self.nx) - K @ H) @ P_pred
        return P_upd
    

    def run_mhe(self, meas: np.ndarray, odometry: list):
        nx = self.nx
        nu = self.nu
        nz = self.nz
        ny = self.ny

        self.z_buffer = np.append(self.z_buffer, meas, axis=1)
        meas = meas.reshape(nx,)


        # Limit horizon length if not enough measurements yet
        M = min(self.M, self.z_buffer.shape[1])

        # Extract last M measurements
        horizon_z = np.zeros((nz, M))
        horizon_u = np.zeros((nu, M))
        horizon_z[:, :M] = self.z_buffer[:, -M:]
        horizon_u[:M] = self.u_buffer[-M:]

        if len(odometry) == 0:
            u = 0
            self.x0_bar = meas
        else:
            u = odometry[-1]
        self.u_buffer.append(u)


        # sneaky cholesky to handle arrival cost without modifying W_0
        L = cholesky(self.P_prior, lower=True) # TODO doesnt wrok :(
        # P = L @ L.T
        # x.T@P^-1@x = x.T@(L@L.T)^-1@x = (L^-1@x).T@I@L^-1@x
        #self.x0_bar = solve_triangular(L, self.x0_bar, lower=True)  # L^-1@x = b -> L@b = x, solve for b


        # shift horizon and update references
        yref_0 = np.zeros(self.ny_0)
        yref_0[:nz] = horizon_z[:, 0]
        yref_0[nz+nx:] = self.x0_bar
        self.solver.set(0, "yref", yref_0)
        self.solver.set(0, "p", horizon_u[:, 0])

        yref = np.zeros(ny)
        for j in range(1, M):
            yref[:nz] = horizon_z[:, j]
            self.solver.set(j, "yref", yref)
            self.solver.set(j, "p", horizon_u[:, j])

        # solve
        if u != 0:
            status = self.solver.solve()
            if status != 0:
                print(f"MHE solver returned status {status}")

        # extract estimate
            x_est = self.solver.get(M, "x")

        # update arrival cost (for next iteration)
            self.x0_bar = self.solver.get(1, "x")
            self.w = self.solver.get(1, "u") # TODO no
            #self.w_buffer = np.append(self.w_buffer, self.w.reshape(nx, 1), axis=1)
        else:
            x_est = meas
            self.x0_bar = x_est
        self.P_prior = self.kalman_update(self.x0_bar.reshape(nx, 1), u) # TODO replace with qr decomposition

        self.x_ests = np.append(self.x_ests, x_est.reshape(nx, 1), axis=1)
        return x_est


class MHE:
    def __init__(self, dyn_mod: dynamic_model, sens_mod: sensor_model, M: int, dt=config.TIME_STEP):
        self.dt = dt  # discretization time

        self.dyn_mod = dyn_mod
        self.sens_mod = sens_mod

        self.M = M  # maximum horizon length
        self.nx = self.dyn_mod.nx  # state dimension
        self.nz = self.sens_mod.nz  # measurement dimension

        # Weights used in cost function, sans arrival cost. Currently set as the inverse of process/measurement noise covariances
        self.W = np.linalg.inv(dyn_mod.Q)
        self.V = np.linalg.inv(sens_mod.R)

        # Buffers for states, inputs and measurements
        self.x_ests = np.empty((self.nx, 0))
        self.z_buffer = np.empty((self.nz, 0))
        self.u_buffer = []

        # Initialize prior. These values are immediately overwritten
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

    def kalman_update(self):
        """Kalman based update to the prior covariance"""
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

        # Extract last N measurements and inputs
        z_seq = self.z_buffer.T[-N:].reshape(N, self.nz, 1)
        u_seq = self.u_buffer[-N:]

        # Initialize trajectory guess with the last optimum for warm start
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
        elif optimizer == 1:
            raise NotImplementedError("Acados optimizer not implemented yet.")

        X_opt = res.x.reshape(N, self.nx, 1)
        self.x_guess = X_opt  # store for warm start next iteration

        # Update arrival cost (last state becomes new prior)
        self.x_prior = X_opt[-1]
        self.kalman_update()  # update prior covariance only

        self.x_ests = np.append(self.x_ests, self.x_prior, axis=1)

        return self.x_prior


# -------------------------
# --- Utility functions ---
# -------------------------


def add_noise(pos, vel):
    z_pos = pos + np.random.normal(0, config.STD_POS)
    z_vel = vel + np.random.normal(0, config.STD_VEL)
    z_meas = np.vstack([z_pos, z_vel])

    return z_meas


def init_estimator(estimator: int):
    """
    2: ekf\n
    3: mhe\n
    4: mhe_acados
    """
    R = lambda v1, v2: np.diag([v1, v2])
    Q = lambda v1, v2, dt: np.array([
        [v1*v2, 0.5*(v1 + v2)*dt**2],
        [0, v2*dt]
        ])
    Q0 = np.diag([1, 1])
    if estimator == 2:
        return EKF(dynamic_model(Q(config.EKF_VAR_PROC_POS, config.EKF_VAR_PROC_VEL, config.TIME_STEP)),
                   sensor_model(R(config.EKF_VAR_MEAS_POS, config.EKF_VAR_MEAS_VEL)))
    if estimator == 3:
        return MHE(dynamic_model(Q(config.MHE_VAR_PROC_POS, config.MHE_VAR_PROC_VEL, config.TIME_STEP)),
                   sensor_model(R, config.MHE_VAR_MEAS_POS, config.MHE_VAR_MEAS_VEL),
                   config.MHE_HORIZON)
    if estimator == 4:
        return MHE_acados(dynamic_model(Q(config.MHE_VAR_PROC_POS, config.MHE_VAR_PROC_VEL, config.TIME_STEP)),
                          sensor_model(R(config.MHE_VAR_MEAS_POS, config.MHE_VAR_MEAS_VEL)),
                          config.MHE_HORIZON, Q0)


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


def run_mhe(mhe: MHE, z: np.ndarray, odometry: list):
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
    3: "MHE",
    4: "MHE_acados"
}


controller_dict = {
    1: "P",
    2: "PPO",
    3: "MPC",
    4: "SMPC",
    5: "TMPC"
}
