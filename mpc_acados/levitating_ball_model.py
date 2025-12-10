'''
from casadi import SX, vertcat
from acados_template import AcadosModel
import config

def levitating_ball_model() -> AcadosModel:

    model_name = 'levitating_ball_ode'

    # Constants
    m = config.BALL_MASS
    g = config.GRAVITY

    # --- Set up states and control ---
    y = SX.sym('y')
    v_y = SX.sym('v_y')

    x = vertcat(y, v_y)

    F = SX.sym('F')
    u = vertcat(F)

    # --- Dynamics ---
    xdot = vertcat(v_y, F/m - g)

    # --- Assign to model ---
    model = AcadosModel()
    model.x = x
    model.u = u
    #model.xdot = xdot
    model.f_expl_expr = xdot
    model.name = model_name

    return model
'''

from casadi import SX, vertcat
from acados_template import AcadosModel
import config

def levitating_ball_model() -> AcadosModel:

    model_name = 'levitating_ball_ode'

    # constants
    m = config.BALL_MASS
    g = config.GRAVITY

    # --- states ---
    y = SX.sym('y')           # height
    v = SX.sym('v')           # velocity
    u_prev = SX.sym('u_prev') # previous control input

    x = vertcat(y, v, u_prev)

    # --- control input ---
    F = SX.sym('F')
    u = vertcat(F)

    # --- dynamics ---
    y_dot = v
    v_dot = F/m - g
    u_prev_dot = 0.0          # hold previous u as state (updated via discrete step)

    xdot = vertcat(y_dot, v_dot, u_prev_dot)

    # --- build model ---
    model = AcadosModel()
    model.x = x
    model.u = u
    model.f_expl_expr = xdot
    model.name = model_name

    return model
