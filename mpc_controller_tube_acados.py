import os
import shutil
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as cs
import config

class MPCControllerTubeAcados:
    """
    Tube MPC implementert med Acados.
    
    Kombinerer en nominell MPC (med strammere begrensninger) og en 
    lineær feedback-regulator (LQR) for å holde systemet innenfor 'tuben'.
    """
    
    def __init__(self, N=config.TUBE_MPC_HORIZON, dt=config.TIME_STEP):
        self.N = N
        self.dt = dt
        
        # --- 1. Parametere for Tube MPC ---
        # Hent stramminger fra config
        self.u_tightening = config.TUBE_MPC_TIGHTING_U
        self.x_tightening = config.TUBE_MPC_TIGHTING_X
        
        # Nominelle grenser (Fysiske grenser - stramming)
        self.lbu_nom = -config.FORCE_MAGNITUDE + self.u_tightening
        self.ubu_nom = config.FORCE_MAGNITUDE - self.u_tightening
        
        # --- 2. Beregn LQR Gain (K) ---
        # Vi trenger K for å beregne feedback: u_tube = K * (x_meas - x_nom)
        # Bruker samme modell som i config/original fil
        A = np.array([[1.0, dt],
                      [0.0, 1.0]])
        B = np.array([[0.0],
                      [dt]])
        
        Q_lqr = np.diag([config.TUBE_MPC_QH, config.TUBE_MPC_QV])
        R_lqr = np.diag([config.TUBE_MPC_R])
        
        # Beregn K en gang ved oppstart
        self.K = self._compute_lqr_gain(A, B, Q_lqr, R_lqr)
        print(f"Tube MPC LQR Gain K: {self.K}")

        # --- 3. Sett opp Acados OCP ---
        self.ocp = self._create_ocp()
        
        # Generer kode og bygg løser
        # Vi bruker et unikt navn for å ikke krasje med standard MPC folderen
        json_file = 'acados_tube_mpc.json'
        
        # Sjekk/Opprett build directory
        build_dir = 'c_generated_code_tube_mpc'
        self.ocp.code_export_directory = build_dir
        
        self.solver = AcadosOcpSolver(self.ocp, json_file=json_file)
        
        # Lagre forrige løsning for warm-start og plotting
        self.last_X_nom = None
        self.last_U_nom = None

    def _compute_lqr_gain(self, A, B, Q, R, max_iters=1000, eps=1e-8):
        """Løser Diskret Riccati-ligning (DARE) iterativt for å finne K."""
        P = Q.copy()
        for _ in range(max_iters):
            # Riccati rekursjon
            inv_term = np.linalg.inv(R + B.T @ P @ B)
            P_next = A.T @ P @ A - (A.T @ P @ B) @ inv_term @ (B.T @ P @ A) + Q
            if np.max(np.abs(P_next - P)) < eps:
                P = P_next
                break
            P = P_next
        
        # K = (R + B'PB)^-1 * B'PA
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        return K

    def _create_ocp(self):
        ocp = AcadosOcp()
        
        # --- Modell ---
        model = AcadosModel()
        model.name = 'ball_maglev_tube'
        
        # Tilstander
        h = cs.SX.sym('h')
        v = cs.SX.sym('v')
        x = cs.vertcat(h, v)
        
        # Pådrag (Nominell kraft)
        F_nom = cs.SX.sym('F_nom')
        u = cs.vertcat(F_nom)
        
        # Dynamikk: x_dot = f(x, u)
        # Nominell modell antar ingen støy
        x_dot = cs.vertcat(v, F_nom - config.GRAVITY)
        
        model.f_expl_expr = x_dot
        model.x = x
        model.u = u
        ocp.model = model
        
        # --- Dimensjoner ---
        ocp.dims.N = self.N
        
        # --- Kostfunksjon (Nominal MPC) ---
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        
        nx = 2
        nu = 1
        
        # Vekter fra config (Tube spesifikke)
        Q_mat = np.diag([config.TUBE_MPC_QH, config.TUBE_MPC_QV])
        R_mat = np.diag([config.TUBE_MPC_R])
        
        # W matrise
        ocp.cost.W = cs.blockcat([[Q_mat, cs.DM.zeros(nx, nu)],
                                  [cs.DM.zeros(nu, nx), R_mat]]).full()
        
        # Terminal kost
        ocp.cost.W_e = Q_mat * config.TUBE_MPC_TERMINAL
        
        # Mapping (Vx, Vu) - Vi straffer avvik fra referanse
        ocp.cost.Vx = np.zeros((nx + nu, nx))
        ocp.cost.Vx[:nx, :] = np.eye(nx)
        ocp.cost.Vu = np.zeros((nx + nu, nu))
        ocp.cost.Vu[nx:, :] = np.eye(nu)
        
        ocp.cost.Vx_e = np.eye(nx)
        
        # Referanser (oppdateres i loopen)
        ocp.cost.yref = np.zeros((nx + nu, ))
        ocp.cost.yref_e = np.zeros((nx, ))
        
        # --- Begrensninger (Constraints) - HER LIGGER MAGIEN I TUBE ---
        
        # 1. Input constraints (Strammet inn)
        ocp.constraints.lbu = np.array([self.lbu_nom])
        ocp.constraints.ubu = np.array([self.ubu_nom])
        ocp.constraints.idxbu = np.array([0])
        
        # 2. State constraints (Strammet inn, f.eks h > 0 + margin)
        # Vi antar h er index 0
        ocp.constraints.lbx = np.array([0.0 + self.x_tightening]) 
        ocp.constraints.ubx = np.array([10000.0]) # Ingen reell øvre grense, men må settes
        ocp.constraints.idxbx = np.array([0])
        
        # 3. Starttilstand (dummy)
        ocp.constraints.x0 = np.array([0.0, 0.0])
        
        # --- Solver Options ---
        ocp.solver_options.tf = self.N * self.dt
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        
        return ocp

    def get_action(self, current_height, current_velocity, target_height, return_trajectory=True):
        """
        1. Løs Nominell MPC -> gir x_nom, u_nom
        2. Beregn feedback -> u_corr = K * (x_meas - x_nom)
        3. Returner u_total = u_nom + u_corr
        """
        
        # 1. Oppdater Referanse
        # Referansevektor yref = [h_ref, v_ref, u_ref]
        # Vi ønsker at nominell u skal være ca null (eller kompensere for g, avhengig av modell)
        u_eq = config.GRAVITY 
        yref = np.array([target_height, 0.0, u_eq])
        yref_e = np.array([target_height, 0.0])
        
        for k in range(self.N):
            self.solver.set(k, "yref", yref)
        self.solver.set(self.N, "yref", yref_e)
        
        # 2. Sett Startbetingelse for Nominell Bane
        x_meas = np.array([current_height, current_velocity])
        # Sikre at det er en flat array (2,)
        x_meas = x_meas.reshape(-1)
        
        self.solver.set(0, "lbx", x_meas)
        self.solver.set(0, "ubx", x_meas)
        
        # 3. Løs
        status = self.solver.solve()
        
        if status != 0:
            # Fallback
            # print(f"Tube MPC Acados failed with status {status}")
            return config.GRAVITY, None, None
            
        # 4. Hent ut Nominell Løsning (x_nom, u_nom)
        u_nom_0 = self.solver.get(0, "u")
        x_nom_0 = self.solver.get(0, "x")
        
        # Hent hele banen for visualisering
        Xn_opt = np.zeros((2, self.N + 1))
        Un_opt = np.zeros((1, self.N))
        for k in range(self.N):
            Xn_opt[:, k] = self.solver.get(k, "x")
            Un_opt[:, k] = self.solver.get(k, "u")
        Xn_opt[:, self.N] = self.solver.get(self.N, "x")
        
        # 5. Beregn Feedback (Tube-leddet)
        # u = u_nom + K * (x_meas - x_nom)
        
        # VIKTIG: Sikre at alt er flatt før subtraksjon for å unngå broadcasting feil
        x_meas_flat = x_meas.reshape(-1)
        x_nom_0_flat = x_nom_0.reshape(-1)
        
        delta_x = x_meas_flat - x_nom_0_flat
        
        # K er shape (1, 2), delta_x er shape (2,) -> u_corr blir shape (1,)
        u_corr = self.K @ delta_x
        
        # Hent ut skalarverdier sikkert
        u_corr_scalar = u_corr.item() if hasattr(u_corr, 'item') else float(u_corr)
        u_nom_scalar = u_nom_0.item() if hasattr(u_nom_0, 'item') else float(u_nom_0)
        
        u_total = u_nom_scalar + u_corr_scalar
        
        # 6. Safety Clipping
        u_applied = np.clip(u_total, -config.FORCE_MAGNITUDE, config.FORCE_MAGNITUDE)
        
        if return_trajectory:
            return float(u_applied), Xn_opt, Un_opt
        else:
            return float(u_applied), None, None

    def sizes(self):
        """Diagnostic info"""
        return (config.TUBE_MPC_QH, config.TUBE_MPC_QV, 
                self.lbu_nom, self.ubu_nom, 
                config.TUBE_MPC_R, config.TUBE_MPC_DELTA_U_MAX)

    def reset_solver(self):
        """Nullstiller solveren."""
        for i in range(self.N + 1):
            self.solver.set(i, "x", np.zeros((2,)))
        for i in range(self.N):
            self.solver.set(i, "u", np.array([config.GRAVITY]))