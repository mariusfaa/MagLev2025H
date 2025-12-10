'''
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from mpc_acados.levitating_ball_model import levitating_ball_model
import config
import os
import shutil

class AcadosMPCController:
 
    def __init__(self, N, T): 
        self.N = N
        self.T = T
        model = levitating_ball_model()

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N

        # --- Solver options ---
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        #ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        #ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.print_level = 0
        ocp.solver_options.tf = self.T

        nx = 2
        nu = 1

        # --- Cost function ---
        # y = [y, vy, F]
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        self.qh = config.STD_MPC_QH
        self.qv = config.STD_MPC_QV
        self.r = config.STD_MPC_R

        # Stage cost
        ocp.cost.W = np.diag([self.qh, self.qv, self.r])

        ocp.cost.Vx = np.zeros((3, nx))
        ocp.cost.Vx[0, 0] = 1.0  # height
        ocp.cost.Vx[1, 1] = 1.0  # velocity

        ocp.cost.Vu = np.zeros((3, nu))
        ocp.cost.Vu[2, 0] = 1.0  # force

        ocp.cost.yref = np.zeros(3)

        # --- Terminal cost ---
        ocp.cost.Vx_e = np.zeros((2, nx))
        ocp.cost.Vx_e[0, 0] = 1.0
        ocp.cost.Vx_e[1, 1] = 1.0

        ocp.cost.W_e = np.diag([self.qh, self.qv])*config.STD_MPC_TERMINAL
        #ocp.cost.Vx_e = np.eye(2)
        ocp.cost.yref_e = np.zeros(2)

        # --- Constraints ---

        # Control bounds
        self.lbu = -1.0*config.FORCE_MAGNITUDE
        self.ubu = config.FORCE_MAGNITUDE
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.lbu = np.array([self.lbu])
        ocp.constraints.ubu = np.array([self.ubu])

        # State bounds: constrain height (>=0)
        ocp.constraints.idxbx = np.array([0])
        ocp.constraints.lbx = np.array([0])
        ocp.constraints.ubx = np.array([2000.0]) #The upper bound on the height is set arbitrarily high

        # Contrain velocity as well:
        #ocp.constraints.idxbx = np.array([0, 1])
        #ocp.constraints.lbx = np.array([0.0, -200.0])
        #ocp.constraints.ubx = np.array([2000.0, 200.0]) #The upper bound on the height is set arbitrarily high

        # Required for using lbx0/ubx0 later:
        ocp.constraints.x0 = np.array([0.0, 0.0])

        # --- Generate solver ---
        # Acados generates C code. We specify a json file and build directory.
        ocp_name = 'ball_acados_mpc'
        ocp.code_export_directory = f'c_generated_code_{ocp_name}'
        
        # Clean up previous build if it exists to avoid conflicts
        if os.path.exists(ocp.code_export_directory):
            shutil.rmtree(ocp.code_export_directory)
            
        self.solver = AcadosOcpSolver(ocp, json_file=f'{ocp_name}.json')

        self.delta_u_max = config.STD_MPC_DELTA_U_MAX #Not fully implemented?

    def get_action(self, current_height, current_velocity, target_height, return_trajectory=True):
        
        # Initial state constraint
        x0 = np.array([current_height, current_velocity])
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # Update stage references
        y_ref = np.array([target_height, 0.0, 0.0])
        for k in range(self.N):
            self.solver.set(k, "yref", y_ref)

        # Terminal refernece
        y_ref_e = np.array([target_height, 0.0])
        # terminal reference = first two elements of yref
        #self.solver.set(self.N, "yref", y_ref[:2])
        self.solver.set(self.N, "yref", y_ref_e)

        # Solve MPC
        status = self.solver.solve()
        if status != 0:
            return 0.0  # fallback

        # Extract optimal control
        u0 = self.solver.get(0, "u")[0]
        #return float(u0)
    
        # Extract Control
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
'''

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from mpc_acados.levitating_ball_model import levitating_ball_model
import config
import os, shutil

class AcadosMPCController:

    def __init__(self, N=config.ACADOS_MPC_HORIZON, T=config.TIME_STEP*config.ACADOS_MPC_HORIZON):

        self.N = N
        self.T = T

        # --- Model & OCP ---
        model = levitating_ball_model()
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = N

        nx = 3   # [y, v, u_prev]
        nu = 1   # [F]

        # =========================
        #  SOLVER OPTIONS
        # =========================
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.print_level = 0
        ocp.solver_options.tf = T

        # =========================
        #  COST FUNCTION
        # =========================
        self.qh = config.ACADOS_MPC_QH
        self.qv = config.ACADOS_MPC_QV
        self.r = config.ACADOS_MPC_R

        # Stage cost: y = [y, vy, F]
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        ocp.cost.W = np.diag([self.qh, self.qv, self.r])
        ocp.cost.W_e = np.diag([self.qh, self.qv]) * config.ACADOS_MPC_TERMINAL

        # Map states and input to cost vector
        ocp.cost.Vx = np.zeros((3, nx))
        ocp.cost.Vx[0, 0] = 1.0  # y
        ocp.cost.Vx[1, 1] = 1.0  # v
        # u_prev column = 0
        ocp.cost.Vu = np.zeros((3, nu))
        ocp.cost.Vu[2, 0] = 1.0  # F

        ocp.cost.Vx_e = np.zeros((2, nx))
        ocp.cost.Vx_e[0, 0] = 1.0
        ocp.cost.Vx_e[1, 1] = 1.0
        # column for u_prev = 0

        # Reference values
        ocp.cost.yref = np.zeros(3)    # matches rows of Vx
        ocp.cost.yref_e = np.zeros(2)  # matches rows of Vx_e

        # =========================
        #  STATE CONSTRAINTS
        # =========================
        ocp.constraints.idxbx = np.array([0])  # only height
        ocp.constraints.lbx = np.array([0.0])
        ocp.constraints.ubx = np.array([2000.0])
        ocp.constraints.x0 = np.zeros(nx)  # [y, v, u_prev]

        # =========================
        #  CONTROL CONSTRAINTS
        # =========================
        Fmax = config.FORCE_MAGNITUDE
        self.lbu = -Fmax
        self.ubu = Fmax
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.lbu = np.array([self.lbu])
        ocp.constraints.ubu = np.array([self.ubu])

        # =========================
        #  DELTA-U CONSTRAINT: -Δu_max ≤ F - u_prev ≤ Δu_max
        # =========================
        self.delta_u_max = config.ACADOS_MPC_DELTA_U_MAX
        ocp.constraints.C = np.array([[0.0, 0.0, -1.0]])  # multiplies state x
        ocp.constraints.D = np.array([[1.0]])             # multiplies input u
        ocp.constraints.lg = np.array([-self.delta_u_max])
        ocp.constraints.ug = np.array([ self.delta_u_max])

        # =========================
        #  BUILD SOLVER
        # =========================
        build_dir = "c_generated_acados_mpc"
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        ocp.code_export_directory = build_dir

        self.solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    # =====================================================================
    #  RUN CONTROLLER
    # =====================================================================
    def get_action(self, y, v, u_prev, target_height, return_trajectory=True):

        x0 = np.array([float(y), float(v), float(u_prev)])
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # Update stage references
        yref = np.array([target_height, 0.0, 0.0])  # [y, vy, F]
        for k in range(self.N):
            self.solver.set(k, "yref", yref)

        # Terminal reference
        yref_e = np.array([target_height, 0.0])
        self.solver.set(self.N, "yref", yref_e)

        # Solve
        status = self.solver.solve()
        if status != 0:
            print("WARNING: ACADOS solver failed:", status)
            return 0.0, None, None

        # Optimal control
        u0 = float(self.solver.get(0, "u"))

        if not return_trajectory:
            return u0, None, None

        # Predicted trajectory
        X = np.zeros((3, self.N+1))
        U = np.zeros((1, self.N))
        for k in range(self.N):
            X[:, k] = self.solver.get(k, "x")
            U[:, k] = self.solver.get(k, "u")
        X[:, self.N] = self.solver.get(self.N, "x")

        return u0, X, U
    
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
