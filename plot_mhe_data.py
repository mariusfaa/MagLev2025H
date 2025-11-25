import numpy as np
import matplotlib.pyplot as plt
import scipy


data = np.load("mhe_data.npz", allow_pickle=True)
ground_truth = data["ground_truth"]
estimated_states = data["estimated_states"]
measurements = data["measurements"]
pos_gt = ground_truth[0]
vel_gt = ground_truth[1]
pos_meas = measurements[0]
vel_meas = measurements[1]

pos_est = estimated_states[0]
vel_est = estimated_states[1]

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(pos_gt, 'r', label='True Ball Height')
plt.plot(pos_meas, 'g.', label='Measured Ball Height')
plt.plot(pos_est, 'b--', label='Estimated Ball Height')
plt.ylabel('Height')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(vel_gt, 'r', label='True Ball Velocity')
plt.plot(vel_meas, 'g.', label='Measured Ball Velocity')
plt.plot(vel_est, 'b--', label='Estimated Ball Velocity')
plt.xlabel('Timestep')
plt.ylabel('Velocity')
plt.legend()
plt.show()

# Compute RMSE for position and velocity
pos_rmse = np.sqrt(np.mean((pos_est - pos_gt)**2))
vel_rmse = np.sqrt(np.mean((vel_est - vel_gt)**2))

print(f"Position RMSE: {pos_rmse:.4f}")
print(f"Velocity RMSE: {vel_rmse:.4f}")

# Plot RMSE over time (cumulative RMSE)
timesteps = len(pos_gt)
pos_rmse_cumulative = np.zeros(timesteps)
vel_rmse_cumulative = np.zeros(timesteps)

for i in range(1, timesteps):
    pos_rmse_cumulative[i] = np.sqrt(np.mean((pos_est[:i+1] - pos_gt[:i+1])**2))
    vel_rmse_cumulative[i] = np.sqrt(np.mean((vel_est[:i+1] - vel_gt[:i+1])**2))

plt.figure(figsize=(12, 8))

# Plot 1: RMSE over time
plt.subplot(2, 2, 1)
plt.plot(pos_rmse_cumulative, 'b-', label=f'Position RMSE (final: {pos_rmse:.4f})')
plt.axhline(y=pos_rmse, color='b', linestyle='--', alpha=0.7)
plt.xlabel('Timestep')
plt.ylabel('RMSE')
plt.title('Position RMSE Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(vel_rmse_cumulative, 'r-', label=f'Velocity RMSE (final: {vel_rmse:.4f})')
plt.axhline(y=vel_rmse, color='r', linestyle='--', alpha=0.7)
plt.xlabel('Timestep')
plt.ylabel('RMSE')
plt.title('Velocity RMSE Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Instantaneous errors
plt.subplot(2, 2, 3)
pos_errors = pos_est - pos_gt
plt.plot(pos_errors, 'b-', label='Position Error')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
plt.axhline(y=pos_rmse, color='b', linestyle='--', alpha=0.7, label=f'RMSE = {pos_rmse:.4f}')
plt.axhline(y=-pos_rmse, color='b', linestyle='--', alpha=0.7)
plt.xlabel('Timestep')
plt.ylabel('Error')
plt.title('Position Estimation Errors')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
vel_errors = vel_est - vel_gt
plt.plot(vel_errors, 'r-', label='Velocity Error')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
plt.axhline(y=vel_rmse, color='r', linestyle='--', alpha=0.7, label=f'RMSE = {vel_rmse:.4f}')
plt.axhline(y=-vel_rmse, color='r', linestyle='--', alpha=0.7)
plt.xlabel('Timestep')
plt.ylabel('Error')
plt.title('Velocity Estimation Errors')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()