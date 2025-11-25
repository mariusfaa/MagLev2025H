import matplotlib.pyplot as plt
from noise import pnoise1
import random

# --- MOCK CONFIGURATION ---
# These mimic your config.py and disturbance settings
TIME_STEP = 0.02        # Simulating 100 updates per second
DURATION = 10.0         # Seconds to simulate
AMPLITUDE = 4.0        # The 'amp' variable in your code
FREQUENCY = 1.0    # The 'freq' variable in your code
OCTAVES = 4             # The 'octaves' variable in your code

def generate_noise_data():
    """
    Simulates the logic inside Ball.apply_force to isolate the noise values.
    """
    # Lists to store data for plotting
    time_points = []
    noise_values = []

    # Simulation variables (replicating self._noise_time)
    current_time = 0.0
    noise_time_cursor = 0.0 
    
    # Loop through the simulation duration
    steps = int(DURATION / TIME_STEP)
    base = random.randint(0, 100)
    for _ in range(steps):
        # 1. Increment time (Replicating: self._noise_time += config.TIME_STEP * freq)
        noise_time_cursor += TIME_STEP * FREQUENCY
        
        # 2. Calculate Noise (Replicating: pnoise1(self._noise_time, octaves=octaves) * amp)
        # Note: pnoise1 returns a value between roughly -1.0 and 1.0
        disturbance = pnoise1(noise_time_cursor, octaves=OCTAVES, base=base) * AMPLITUDE
        
        # 3. Store data
        time_points.append(current_time)
        noise_values.append(disturbance)
        
        current_time += TIME_STEP

    return time_points, noise_values

def plot_data(x, y):
    """Sets up the Matplotlib graph."""
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=f'Perlin Noise (Octaves={OCTAVES})', color='royalblue', linewidth=1.5)
    
    # Formatting
    plt.title(f'Disturbance Force Over Time\n(Freq: {FREQUENCY}, Amp: {AMPLITUDE})')
    plt.xlabel('Simulation Time (seconds)')
    plt.ylabel('Force Applied (Disturbance)')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--') # Zero line
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Generating noise data...")
    times, values = generate_noise_data()
    print(f"Plotting {len(values)} data points.")
    plot_data(times, values)