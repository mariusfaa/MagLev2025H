import numpy as np
import matplotlib.pyplot as plt

data = np.load("mpc_data.npz")
positions = data["positions"]
forces = data["forces"]
trajectories = data.get("trajectories", None)
pred_controls = data.get("controls", None)

plt.figure(figsize=(10, 5))
plt.suptitle(f"qu = {data['qu']}, qx = {data['qx']}, lbu = {data['lbu']}, ubu = {data['ubu']}, r = {data['r']}, N = {data['N']}, delta_u_max = {data['delta_u_max']}")
plt.subplot(2, 1, 1)
plt.plot(positions, label='Ball Height')
plt.plot([data['ref']]*len(positions), 'r--', label='Target Height')
# Overlay predicted trajectories (height) at each timestep
if trajectories is not None:
	try:
		K = 10  # plot only every Kth horizon
		for i, traj in enumerate(trajectories):
			if i % K != 0 or traj is None:
				continue
			traj = np.array(traj)
			heights = traj[0, :]
			# plot predicted horizon starting at current timestep
			x_idx = np.arange(i, i + len(heights))
			plt.plot(x_idx, heights, color='gray', alpha=0.35)
	except Exception:
		# Fallback for older numpy save formats
		pass
plt.ylabel('Height')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(forces, label='Control Input (Force)', color='orange')
# Optionally plot predicted control horizons as faint lines
if pred_controls is not None:
	try:
		K = 5
		for i, u_traj in enumerate(pred_controls):
			if i % K != 0 or u_traj is None: 
				continue
			u_traj = np.array(u_traj)
			x_idx = np.arange(i, i + len(u_traj))
			plt.plot(x_idx, u_traj, color='orange', alpha=0.35, linestyle='--')
	except Exception:
		pass
plt.xlabel('Timestep')
plt.ylabel('Force')
plt.legend()

# plt.tight_layout()
plt.show()