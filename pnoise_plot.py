import matplotlib.pyplot as plt
from noise import pnoise1
import random
import csv

# --- MOCK CONFIGURATION ---
# These mimic your config.py and disturbance settings
TIME_STEP = 0.02        # Simulating 100 updates per second
DURATION = 30.0         # Seconds to simulate
AMPLITUDE = 4.0         # The 'amp' variable in your code
FREQUENCY = 1.0         # The 'freq' variable in your code
OCTAVES = 2             # The 'octaves' variable in your code

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
        # 1. Increment time
        noise_time_cursor += TIME_STEP * FREQUENCY
        
        # 2. Calculate Noise
        disturbance = pnoise1(noise_time_cursor, octaves=OCTAVES, base=base) * AMPLITUDE
        
        # 3. Store data
        time_points.append(current_time)
        noise_values.append(disturbance)
        
        current_time += TIME_STEP

    return time_points, noise_values

def save_to_csv(filename, times, values):
    """
    Lagrer tid og støyverdier til en CSV-fil.
    """
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time', 'Disturbance'])
            for t, v in zip(times, values):
                writer.writerow([f"{t:.4f}", f"{v:.4f}"])
        print(f"Suksess! Data lagret til filen: {filename}")
    except IOError as e:
        print(f"Feil ved lagring av fil: {e}")

def plot_data(x, y):
    """Sets up the Matplotlib graph and saves it to a file."""
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=f'Perlin Noise (Octaves={OCTAVES})', color='royalblue', linewidth=1.5)
    
    # Formatting
    plt.title(f'Disturbance Force Over Time\n(Freq: {FREQUENCY}, Amp: {AMPLITUDE})')
    plt.xlabel('Simulation Time (seconds)')
    plt.ylabel('Force Applied (Disturbance)')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--') # Zero line
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # --- CHANGE START ---
    # Save the figure to a file instead of just showing it
    filename = 'pnoise_plot.png'
    plt.savefig(filename, dpi=300)
    print(f"Plot saved successfully to: {filename}")
    # --- CHANGE END ---

if __name__ == "__main__":
    print("Generating noise data...")
    times, values = generate_noise_data()
    
    # Save to CSV
    csv_filename = "pnoise_data.csv"
    save_to_csv(csv_filename, times, values)
    
    print(f"Plotting {len(values)} data points.")
    plot_data(times, values)