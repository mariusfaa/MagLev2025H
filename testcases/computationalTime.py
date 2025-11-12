import time
import numpy as np
import sys
import os

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def calculate_average_computational_time(controller, num_states=100):
    """
    Calculates the average computational time of a given controller over n viable states.

    Args:
        controller: The controller object with a get_action(height, velocity) method.
                    as input and return a control signal.
        num_states (int): The number of viable states to test.

    Returns:
        float: The average computational time in seconds.
    """
    # Generate a whole array of viable states outside the loop.
    # This ensures that state generation time is not part of the measurement.
    # The states are based on the simulation's physical boundaries.
    min_pos = config.GROUND_HEIGHT + config.BALL_RADIUS
    max_pos = config.SCREEN_HEIGHT - config.BALL_RADIUS
    
    # A wider range of possible states
    positions = np.random.uniform(min_pos, max_pos, num_states)
    velocities = np.random.uniform(-50, 50, num_states) # Wider velocity range
    states = list(zip(positions, velocities))

    # Time the entire loop where only the controller action is calculated.
    start_time = time.perf_counter()
    for ball_position, ball_velocity in states:
        _ = controller.get_action(ball_position, ball_velocity)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time

    return total_time / num_states if num_states > 0 else 0

if __name__ == '__main__':
    # Example usage with PController from the project
    from p_controller import PController
    p_control = PController(kp=0.8)
    print("Calculating average computational time for PController...")
    avg_time = calculate_average_computational_time(p_control, num_states=10000)
    print(f"Average computational time: {avg_time * 1e6:.2f} microseconds")
