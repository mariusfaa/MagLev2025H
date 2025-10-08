import numpy as np
import matplotlib.pyplot as plt

data = np.load("mpc_data.npz")
positions = data["positions"]
forces = data["forces"]

plt.figure(figsize=(10, 5))
plt.suptitle(f"qu = {data['qu']}, qx = {data['qx']}, lbu = {data['lbu']}, ubu = {data['ubu']}, r = {data['r']}, N = {data['N']}, delta_u_max = {data['delta_u_max']}")
plt.subplot(2, 1, 1)
plt.plot(positions, label='Ball Height')
plt.plot([data['ref']]*len(positions), 'r--', label='Target Height')
plt.ylabel('Height')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(forces, label='Control Input (Force)', color='orange')
plt.xlabel('Timestep')
plt.ylabel('Force')
plt.legend()

# plt.tight_layout()
plt.show()