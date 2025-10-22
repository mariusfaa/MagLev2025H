import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
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