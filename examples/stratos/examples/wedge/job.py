import matplotlib.pyplot as plt
from stratos import Stratos
import numpy as np

# -----------------------------------------------------------------------------
# Simulation Setup
# -----------------------------------------------------------------------------
st = Stratos('air-5sp.gas', 'air-5sp.chem', 'air-5sp.kin', 0.1, 1000, 6075.1, {"N2":0.767, "O2":0.233}, 365.50, rho1=0.02322)

angles = [5, 10, 15, 20, 25, 30, 35, 40]

# -----------------------------------------------------------------------------
# Loop through angles
# -----------------------------------------------------------------------------
results = {}

for angle in angles:
    print(f"Running simulation for {angle} degrees...")
    results[angle], _ = st.oblique(np.deg2rad(angle), verbose=False)

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
for angle in angles:
    strat = results[angle]
    plt.plot(strat['x_p'], strat['T_tr'], label=fr'$\theta$ = {angle}$^\circ$')

plt.xlabel("Distance along body, m")
plt.ylabel("Temperature, K")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()
plt.savefig('wedge_temperature.png', dpi=300)

plt.figure(figsize=(7, 5))
for angle in angles:
    strat = results[angle]
    plt.plot(strat['x_p'], strat['rho'], label=fr'$\theta$ = {angle}$^\circ$')

plt.xlabel("Distance along body, m")
plt.ylabel(r"Density, kg/m$^3$")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()
plt.savefig('wedge_density.png', dpi=300)
