import numpy as np
import config

class PController:
    """A simple Proportional (P) controller."""
    def __init__(self, kp):
        """Initializes the controller with a proportional gain (kp)."""
        self.kp = kp

    def get_action(self, current_height, target_height, velocity = 0):
        """
        Calculates the required force based on the proportional error
        between the current height and the target height.
        """
        error = target_height - current_height
        force = self.kp * error - velocity*0.001

        return np.clip(force, -config.FORCE_MAGNITUDE, config.FORCE_MAGNITUDE)
