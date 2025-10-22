import numpy as np
from filter import gaussian
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy


data = np.load("ekf_data.npz", allow_pickle=True)
ground_truth = data["ground_truth"]
estimated_states = data["estimated_states"]
estimated_measurements = data["estimated_measurements"]
measurements = data["measurements"]
pos_gt = ground_truth[0]
vel_gt = ground_truth[1]
pos_meas = measurements[0]
vel_meas = measurements[1]

state_means = np.column_stack([g.mean for g in estimated_states])
meas_means = np.column_stack([g.mean for g in estimated_measurements])

pos_est = state_means[0]
vel_est = state_means[1]
pos_meas_est = meas_means[0]
vel_meas_est = meas_means[1]

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

# Compute NIS and NEES and plot on a logarithmic scale with 95% CI
plt.figure(figsize=(10, 6))

# prepare arrays
timesteps = len(pos_meas)
nis_values = np.empty(timesteps)
nees_values = np.empty(timesteps)

for i in range(timesteps):
    z = np.vstack([pos_meas[i], vel_meas[i]])
    nis_values[i] = estimated_measurements[i].mahalanobis_distance(z)

    x_gt = np.vstack([pos_gt[i], vel_gt[i]])
    nees_values[i] = estimated_states[i].mahalanobis_distance(x_gt)

# degrees of freedom
# measurement dim (m) and state dim (n)
m = z.size if 'z' in locals() else 2
n = x_gt.size if 'x_gt' in locals() else 2

# 95% confidence interval from chi-square
nis_ci = scipy.stats.chi2.ppf([0.025, 0.975], df=m)
nees_ci = scipy.stats.chi2.ppf([0.025, 0.975], df=n)

eps = 1e-12

# NIS plot (log scale)
plt.subplot(2, 1, 1)
plt.semilogy(np.maximum(nis_values, eps), label='NIS')
inside_nis = np.mean((nis_values >= nis_ci[0]) & (nis_values <= nis_ci[1])) * 100.0
plt.hlines(nis_ci[0], 0, timesteps - 1, colors='C1', linestyles='--', label=rf'$\chi^2$' + f' 95% CI' + f" ({inside_nis:.1f}% inside)")
plt.hlines(nis_ci[1], 0, timesteps - 1, colors='C1', linestyles='--')
plt.title('NIS')
plt.ylabel('NIS')
plt.legend(loc='upper right')

# NEES plot (log scale)
plt.subplot(2, 1, 2)
plt.semilogy(np.maximum(nees_values, eps), label='NEES')
inside_nees = np.mean((nees_values >= nees_ci[0]) & (nees_values <= nees_ci[1])) * 100.0
plt.hlines(nees_ci[0], 0, timesteps - 1, colors='C1', linestyles='--', label=rf'$\chi^2$' + f' 95% CI' + f" ({inside_nees:.1f}% inside)")
plt.hlines(nees_ci[1], 0, timesteps - 1, colors='C1', linestyles='--')
plt.title('NEES')
plt.xlabel('Timestep')
plt.ylabel('NEES')
plt.legend(loc='upper right')

plt.show()