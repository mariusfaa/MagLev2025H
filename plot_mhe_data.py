import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from statsmodels.graphics.tsaplots import plot_acf



avg_runtime = lambda x: sum(x)/float(len(x))


data = pd.read_csv("filter_results/erk/mhe_refconstant_freq1_slope5_octave4_horizon100.csv")
runtimes = data["runtimes"].values
runtimes_kalman = data["runtimes_kalman"].values

avg_total_runtime = avg_runtime(runtimes)
avg_kalman_runtime = avg_runtime(runtimes_kalman)
avg_opt_runtime = avg_total_runtime - avg_kalman_runtime

print(f"Average runtime: {avg_total_runtime}")
print(f"Average runtime kalman: {avg_kalman_runtime}")
print(f"Average optimizer runtime: {avg_opt_runtime}")

pos_gt = data["pos_gt"].values
vel_gt = data["vel_gt"].values
pos_meas = data["pos_meas"].values
vel_meas = data["vel_meas"].values
pos_est = data["pos_ests"].values
vel_est = data["vel_ests"].values
disturbance = data["disturbance"].values

plt.figure(figsize=(10, 5))
plt.subplot(3, 1, 1)
plt.plot(pos_meas, 'g.', label='Measured Ball Height [m]')
plt.plot(pos_est, 'b--', label='Estimated Ball Height [m]')
plt.plot(pos_gt, 'r', label='True Ball Height [m]')
plt.ylabel('Height [m]')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(vel_meas, 'g.', label='Measured Ball Velocity [m/s]')
plt.plot(vel_est, 'b--', label='Estimated Ball Velocity [m/s]')
plt.plot(vel_gt, 'r', label='True Ball Velocity [m/s]')
plt.xlabel('Timestep')
plt.ylabel('Velocity [m/s]')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(disturbance, 'r', label='Perlin disturbance')
plt.xlabel('Timestep')
plt.ylabel('Force [N]')
plt.legend()
plt.show()
'''
plt.figure(figsize=(10, 5))
plot_acf(disturbance, lags=50, bartlett_confint=False, alpha=None, ax=plt.gca())
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.savefig('plots/acf.png')
plt.show()
'''
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