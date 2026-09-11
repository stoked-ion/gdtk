from crbl import BoundaryLayer, config
import matplotlib.pyplot as plt
import numpy as np

# Setup solver.
config.gas_model_file = "air-7sp.gas"
config.chem_model_file = "air-7sp.chem"
config.reacting = True
config.n_x_cells = 500
config.n_y_cells = 500
config.init_gas_model()

# Inflow conditions.
u1 = 8601
p1 = 530
T1 = 1936
massf1 = [0.767, 0.233, 0, 0, 0, 0, 0]

x = np.linspace(0.0001, 0.13, config.n_x_cells)
P_e = p1 * (1 + 0.05 / np.sqrt(x))
u_e = np.full(config.n_x_cells, u1)
T_e = np.full(config.n_x_cells, T1)

mass_f_e = np.full((config.n_x_cells, config.gmodel.n_species), massf1)

# Initialise CRBL.
bl = BoundaryLayer(x_arr=x, u_e_arr=u_e, T_e_arr=T_e, P_e_arr=P_e, massf_e_arr=mass_f_e)

eta, y, bl_height, delta_star, U, massf, rho, T, k, mu, tau_w, q_c, q_d = bl.run()

# Plotting
plt.plot(x, delta_star, label="d*")
plt.plot(x, bl_height, label="d")
plt.xlabel("x, m")
plt.ylabel("y, m")
plt.legend()
plt.tight_layout()
plt.savefig("bl.png", dpi=300)
