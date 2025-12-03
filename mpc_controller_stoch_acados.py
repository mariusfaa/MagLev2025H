import os
import shutil
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as cs
import config

class MPCControllerStochasticAcados:
    """
    A Stochastic MPC Controller utilizing Acados by simulating an ensemble 
    of systems (samples) simultaneously to optimize expected cost.
    """
    
    def __init__(self, N=config.STOCHASTIC_MPC_HORIZON, dt=config.TIME_STEP, num_samples=config.STOCHASTIC_MPC_SAMPLES):
        self.N = N
        self.dt = dt
        self.num_samples = num_samples
        
        # 1. Define the Ensemble Acados Model
        model = self._create_ensemble_model()

        # 2. Define the Optimal Control Problem (OCP)
        ocp = AcadosOcp()
        ocp.model = model
        
        # Dimensions
        ocp.dims.N = N
        nx = model.x.size()[0] # (2 * num_samples)
        nu = model.u.size()[0] # 1 (shared force)
        np_param = model.p.size()[0] # (2 * num_samples) - Noise parameters

        # 3. Cost Function (Linear Least Squares)
        # We want to minimize average error: sum((x_i - ref)^2) / num_samples
        # This is equivalent to minimizing sum((x_i - ref)^2) with weights scaled by 1/num_samples
        
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        # Scale weights by number of samples to get the average behavior
        qx_scaled = config.STOCHASTIC_MPC_QH / num_samples
        qv_scaled = config.STOCHASTIC_MPC_QV / num_samples
        r_scaled = config.STOCHASTIC_MPC_R   # Control effort is shared, no need to divide by samples usually, but depends on intent. 
                                             # Usually we penalize the single force applied.
        
        # Construct Vx (State mapping) and Vu (Control mapping)
        # We map all states to a cost vector y of size (2 * num_samples + 1)
        # y = [h1, v1, h2, v2, ..., hN, vN, u]
        
        ny = 2 * num_samples + 1
        ny_e = 2 * num_samples

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vu = np.zeros((ny, nu))
        
        # Fill diagonal blocks for each sample
        W_diag = []
        for i in range(num_samples):
            # Map state i to output i
            idx_start = i * 2
            ocp.cost.Vx[idx_start, idx_start] = 1.0     # height
            ocp.cost.Vx[idx_start+1, idx_start+1] = 1.0 # velocity
            
            # Add weights
            W_diag.append(qx_scaled)
            W_diag.append(qv_scaled)
            
        # Map control to the last element of y
        ocp.cost.Vu[ny-1, 0] = 1.0
        W_diag.append(r_scaled)
        
        ocp.cost.W = np.diag(W_diag)
        ocp.cost.yref = np.zeros(ny)

        # --- Terminal Cost ---
        ocp.cost.Vx_e = np.zeros((ny_e, nx))
        W_e_diag = []
        terminal_factor = config.STOCHASTIC_MPC_TERMINAL
        
        for i in range(num_samples):
            idx_start = i * 2
            ocp.cost.Vx_e[idx_start, idx_start] = 1.0
            ocp.cost.Vx_e[idx_start+1, idx_start+1] = 1.0
            
            W_e_diag.append(qx_scaled * terminal_factor)
            W_e_diag.append(qv_scaled * terminal_factor)
            
        ocp.cost.W_e = np.diag(W_e_diag)
        ocp.cost.yref_e = np.zeros(ny_e)

        # 4. Constraints
        self.lbu = config.FORCE_MAGNITUDE * -1
        self.ubu = config.FORCE_MAGNITUDE
        ocp.constraints.lbu = np.array([self.lbu])
        ocp.constraints.ubu = np.array([self.ubu])
        ocp.constraints.idxbu = np.array([0])

        # State bounds: Height >= 0 for ALL samples
        # Because x is stacked [h1, v1, h2, v2...], height indices are 0, 2, 4...
        h_indices = [i*2 for i in range(num_samples)]
        
        ocp.constraints.lbx = np.zeros(num_samples) # All heights >= 0
        ocp.constraints.ubx = np.full(num_samples, 10000.0)
        ocp.constraints.idxbx = np.array(h_indices)

        ocp.constraints.x0 = np.zeros(nx)
        
        # Parameters (Noise)
        ocp.parameter_values = np.zeros(np_param)

        # 5. Solver Options
        ocp.solver_options.tf = N * dt
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        
        # 6. Generate Solver
        ocp_name = 'ball_stoch_mpc'
        ocp.code_export_directory = f'c_generated_code_{ocp_name}'
        
        if os.path.exists(ocp.code_export_directory):
            shutil.rmtree(ocp.code_export_directory)
            
        self.solver = AcadosOcpSolver(ocp, json_file=f'{ocp_name}.json')
        
        # Prepare noise samples (Pre-generating as in your original logic)
        self.noise_std = config.STOCHASTIC_MPC_NOISE_STD
        # Shape: (N, 2 * num_samples) - Noise for each step and each sample state
        self.noise_samples_trajectory = np.random.normal(0, self.noise_std, (N, 2 * num_samples))

    def _create_ensemble_model(self):
        """Creates a model stacking 'num_samples' ball dynamics."""
        model = AcadosModel()
        model.name = 'ball_ensemble_dynamics'

        # Control (Shared)
        F = cs.SX.sym('F')
        u = cs.vertcat(F)
        
        x_list = []
        x_dot_list = []
        p_list = [] # Parameters for noise

        # Define dynamics for each sample
        for i in range(self.num_samples):
            # States
            h = cs.SX.sym(f'h_{i}')
            v = cs.SX.sym(f'v_{i}')
            x_list.extend([h, v])
            
            # Noise Parameters (treated as external input/param in prediction)
            w_h = cs.SX.sym(f'w_h_{i}') # process noise on pos (if any)
            w_v = cs.SX.sym(f'w_v_{i}') # process noise on vel
            p_list.extend([w_h, w_v])

            # Dynamics x_dot = f(x, u, p)
            # Assuming noise is additive constant over the step for integration
            # h_dot = v
            # v_dot = (F/m - g) + w_v
            # Note: Your original code added noise directly to next state. 
            # In ODE, we add it as a disturbance term.
            
            x_dot_list.append(v + w_h) 
            x_dot_list.append(F - config.GRAVITY + w_v)

        model.f_expl_expr = cs.vertcat(*x_dot_list)
        model.x = cs.vertcat(*x_list)
        model.u = u
        model.p = cs.vertcat(*p_list)
        
        return model

    def get_action(self, current_height, current_velocity, target_height, return_trajectory=True):
        
        # 1. Set Initial State
        # Ensure inputs are standard floats to avoid Autograd/Numpy shape issues
        h = float(current_height)
        v = float(current_velocity)
        
        # All samples start at the same true state
        x0_single = np.array([h, v], dtype=np.float64)
        x0_ensemble = np.tile(x0_single, self.num_samples)
        
        # Acados requires explicit contiguous arrays for C-interface
        x0_ensemble = np.ascontiguousarray(x0_ensemble)
        
        self.solver.set(0, "lbx", x0_ensemble)
        self.solver.set(0, "ubx", x0_ensemble)

        # 2. Update References and Noise Parameters
        # Reference vector y = [h_ref, 0, h_ref, 0 ... , 0]
        y_ref_single = [target_height, 0.0]
        y_ref = np.concatenate([np.tile(y_ref_single, self.num_samples), [0.0]]) # + control ref
        
        y_ref_e = np.tile(y_ref_single, self.num_samples)

        for k in range(self.N):
            self.solver.set(k, "yref", y_ref)
            # Set noise parameter for this step
            # Note: In real stochastic MPC, we might re-sample this every step or keep it fixed
            # Here we just cycle or use random noise. 
            # For strict equivalence to your code, we use pre-generated noise or re-generate.
            # Let's re-sample for better stochastic behavior or use a slice of pre-gen.
            
            # Simple approach: Randomize noise parameter at each call (or keep fixed trajectory)
            # Using the pre-calculated noise block corresponding to step k
            p_val = self.noise_samples_trajectory[k, :] 
            self.solver.set(k, "p", p_val)

        self.solver.set(self.N, "yref", y_ref_e)

        # 3. Solve
        status = self.solver.solve()
        
        if status != 0:
            return config.GRAVITY, None, None

        optimal_force = self.solver.get(0, "u")[0]

        # 4. Extract Trajectory (Mean of samples for visualization)
        if return_trajectory:
            # Get full stacked state
            predicted_X_ensemble = np.zeros((2 * self.num_samples, self.N + 1))
            for k in range(self.N + 1):
                predicted_X_ensemble[:, k] = self.solver.get(k, "x")
            
            predicted_U = np.zeros((1, self.N))
            for k in range(self.N):
                predicted_U[:, k] = self.solver.get(k, "u")
                
            # Compute Mean Trajectory for plotting
            # Reshape to (num_samples, 2, N+1)
            X_reshaped = predicted_X_ensemble.reshape(self.num_samples, 2, self.N + 1)
            predicted_X_mean = np.mean(X_reshaped, axis=0)

            return optimal_force, predicted_X_mean, predicted_U
        
        return optimal_force, None, None

    def sizes(self):
        return config.STOCHASTIC_MPC_QH, config.STOCHASTIC_MPC_QV, self.lbu, self.ubu, config.STOCHASTIC_MPC_R, config.STD_MPC_DELTA_U_MAX