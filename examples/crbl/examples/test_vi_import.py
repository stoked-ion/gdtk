from crbl import ViscousInteraction, config
import matplotlib.pyplot as plt
import numpy as np
import eqc

# Setup Solver.
config.gas_model_file = "air-7sp.gas"
config.chem_model_file = "air-7sp.chem"
config.reacting = True
config.n_x_cells = 200
config.n_y_cells = 100
config.export = False
config.catalytic = True
config.omega = 0.5
config.init_gas_model()

# Inflow conditions.
u1 = 8601
p1 = 530
T1 = 1936
massf1 = [0.767, 0.233, 0, 0, 0, 0, 0]

# Convert to eq inflow.
eq = eqc.EqCalculator(config.gmodel.species_names)
Y0 = np.array(massf1)
X0 = eq.YtoX(Y0)  
X1 = eq.pt(p=p1, T=T1, Xs0=X0)
Y1 = eq.XtoY(X1).tolist()

# Initialise CRBL.
vi = ViscousInteraction(u=u1, p=p1, T=T1, massf=Y1, x_end=0.13)

x, d_star, P_e, tau_w, q_c, q_d  = vi.run()

# Plot the result.
plt.plot(x, P_e)
plt.xlabel("x, m")
plt.ylabel("p, Pa")
plt.savefig("vi.png", dpi=300)