import os
import shutil
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as cs
import config

class MPCControllerACADOS:
    """A high-performance MPC Controller using Acados."""
    
    def __init__(self, N=config.STD_MPC_HORIZON, dt=config.TIME_STEP):
        self.N = N
        self.dt = dt
        
        # 1. Define the Acados Model (Dynamics & Symbols)
        model = self._create_model()

        # 2. Define the Optimal Control Problem (OCP)
        ocp = AcadosOcp()
        ocp.model = model
        
        # Dimensions
        ocp.dims.N = N
        nx = model.x.size()[0]
        nu = model.u.size()[0]

        # 3. Cost Function (Linear Least Squares)
        # We target y_ref = [h_ref, 0, 0] (height, vel, force)
        # Cost structure: || Vx*x + Vu*u - y_ref ||^2_W
        
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        # Map state x and input u to cost vector y
        ocp.cost.Vx = np.zeros((3, nx))
        ocp.cost.Vx[0, 0] = 1.0  # height
        ocp.cost.Vx[1, 1] = 1.0  # velocity
        
        ocp.cost.Vu = np.zeros((3, nu))
        ocp.cost.Vu[2, 0] = 1.0  # force

        # Weight Matrix W
        self.qh = config.STD_MPC_QH
        self.qv = config.STD_MPC_QV
        self.r = config.STD_MPC_R
        
        # Q matrix [qh, qv, r]
        W = np.diag([self.qh, self.qv, self.r])
        ocp.cost.W = W

        # Initial Reference (will be updated in loop)
        ocp.cost.yref = np.array([0.0, 0.0, 0.0])

        # --- Terminal Cost ---
        # Only penalize state at the end
        ocp.cost.Vx_e = np.zeros((2, nx))
        ocp.cost.Vx_e[0, 0] = 1.0
        ocp.cost.Vx_e[1, 1] = 1.0

        # Terminal Weight (scaled by config factor)
        W_e = np.diag([self.qh, self.qv]) * config.STD_MPC_TERMINAL
        ocp.cost.W_e = W_e
        ocp.cost.yref_e = np.array([0.0, 0.0])

        # 4. Constraints
        # Control bounds
        self.lbu = config.FORCE_MAGNITUDE * -1
        self.ubu = config.FORCE_MAGNITUDE
        ocp.constraints.lbu = np.array([self.lbu])
        ocp.constraints.ubu = np.array([self.ubu])
        ocp.constraints.idxbu = np.array([0]) # Index of u to constrain

        # State bounds (Height >= 0)
        ocp.constraints.lbx = np.array([0.0])
        ocp.constraints.ubx = np.array([10000.0]) # Arbitrary large upper bound
        ocp.constraints.idxbx = np.array([0])     # Index of x to constrain (height)

        # Initial state (dummy value, updated at runtime)
        ocp.constraints.x0 = np.array([0.0, 0.0])

        # 5. Solver Options
        ocp.solver_options.tf = N * dt
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # Fast QP solver
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # Real-Time Iteration (fastest)
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK' # Explicit Runge-Kutta
        ocp.solver_options.print_level = 0
        
        # 6. Generate Solver
        # Acados generates C code. We specify a json file and build directory.
        ocp_name = 'ball_mpc'
        ocp.code_export_directory = f'c_generated_code_{ocp_name}'
        
        # Clean up previous build if it exists to avoid conflicts
        if os.path.exists(ocp.code_export_directory):
            shutil.rmtree(ocp.code_export_directory)
            
        self.solver = AcadosOcpSolver(ocp, json_file=f'{ocp_name}.json')
        
        # Store params for sizes()
        self.delta_u_max = config.STD_MPC_DELTA_U_MAX 

    def _create_model(self):
        """Defines the ODE using CasADi."""
        model = AcadosModel()
        model.name = 'ball_dynamics'

        # States
        h = cs.SX.sym('h')
        v = cs.SX.sym('v')
        x = cs.vertcat(h, v)

        # Controls
        F = cs.SX.sym('F')
        u = cs.vertcat(F)

        # Dynamics (Continuous time: x_dot = f(x,u))
        # h_dot = v
        # v_dot = u - g
        x_dot = cs.vertcat(v, F - config.GRAVITY)

        model.f_expl_expr = x_dot
        model.x = x
        model.u = u
        
        return model

    def get_action(self, current_height, current_velocity, target_height, return_trajectory=True):
        """Computes optimal control."""
        
        # 1. Set Initial State Constraint
        x0 = np.array([current_height, current_velocity])
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # 2. Update Reference (Cost function targets)
        # y_ref = [target_height, 0, 0] (we assume 0 force reference usually, or gravity comp)
        y_ref = np.array([target_height, 0.0, 0.0])
        
        for k in range(self.N):
            self.solver.set(k, "yref", y_ref)
        
        # Terminal reference
        y_ref_e = np.array([target_height, 0.0])
        self.solver.set(self.N, "yref", y_ref_e)

        # 3. Solve
        status = self.solver.solve()

        if status != 0:
            # Fallback if solver fails (rare in SQP_RTI)
            # print(f"Acados returned status {status}")
            return config.GRAVITY, None, None

        # 4. Extract Control and Trajectories
        optimal_force = self.solver.get(0, "u")[0]
        
        # Extract predicted trajectory for visualization
        predicted_X = np.zeros((2, self.N + 1))
        predicted_U = np.zeros((1, self.N))
        
        for k in range(self.N):
            predicted_X[:, k] = self.solver.get(k, "x")
            predicted_U[:, k] = self.solver.get(k, "u")
        predicted_X[:, self.N] = self.solver.get(self.N, "x")

        if return_trajectory:
            return optimal_force, predicted_X, predicted_U
        else:
            return optimal_force, None, None

    def sizes(self):
        """Returns the sizes of the state and action spaces."""
        return self.qh, self.qv, self.lbu, self.ubu, self.r, self.delta_u_max
    
    def reset_solver(self):
        """Resets the solver internal variables to zero/initial guess."""
        # Reset state and input guesses for the whole horizon
        for i in range(self.N + 1):
            self.solver.set(i, "x", np.zeros((2,))) 
        for i in range(self.N):
            self.solver.set(i, "u", np.array([config.GRAVITY])) # Guess gravity compensation