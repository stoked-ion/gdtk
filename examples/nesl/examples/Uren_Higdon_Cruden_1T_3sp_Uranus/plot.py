import matplotlib.pyplot as plt
import numpy as np

nesl = np.loadtxt("uren.dat", skiprows=1, delimiter=",")
eilmer = np.loadtxt("eilmer.csv", skiprows=1)

x_n, T_n, p_n, rho_n, u_n = nesl[:, 0], nesl[:, 3], nesl[:, 4], nesl[:, 2], nesl[:, 1]
x_e, T_e, p_e, rho_e, u_e = eilmer[:, 0], eilmer[:, 16], eilmer[:, 6], eilmer[:, 3], eilmer[:, 4]

x_e = x_e - x_e[-1] + x_n[-1]

fig, axs = plt.subplots(2, 2, figsize=(8, 7))

axs[0, 0].plot(x_e, T_e, label="eilmer")
axs[0, 0].plot(x_n, T_n, label="nesl")
axs[0, 0].set_ylabel("T, K")
axs[0, 0].legend()

axs[0, 1].plot(x_e, p_e)
axs[0, 1].plot(x_n, p_n)
axs[0, 1].set_ylabel("p, Pa")

axs[1, 0].plot(x_e, rho_e)
axs[1, 0].plot(x_n, rho_n)
axs[1, 0].set_ylabel("rho, kg/m^3")
axs[1, 0].set_xlabel("x, m")

axs[1, 1].plot(x_e, u_e)
axs[1, 1].plot(x_n, u_n)
axs[1, 1].set_ylabel("u, m/s")
axs[1, 1].set_xlabel("x, m")    

plt.tight_layout()
plt.savefig("prim.png")

plt.show()