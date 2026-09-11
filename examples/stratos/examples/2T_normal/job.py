from normalshock import normal_shock, GasState
import matplotlib.pyplot as plt
from stratos import Stratos
import numpy as np
import eqc

# -----------------------------------------------------------------------------
# Simulation Setup
# -----------------------------------------------------------------------------
st = Stratos('air-5sp.gas', 'air-5sp.chem', 'air-5sp.kin', 0.2, 100000, 4270, {"N2":0.802, "O2":0.198}, 300, 133.3)
strat = st.normal()

# -----------------------------------------------------------------------------
# Load Reference Data
# -----------------------------------------------------------------------------
lmr_data = np.loadtxt('lmr.csv', skiprows=1)

lmr_x = lmr_data[:, 0]
lmr_T_tr = lmr_data[:, 16]
lmr_T_ve = lmr_data[:, 18]
lmr_p = lmr_data[:, 6] 
lmr_rho = lmr_data[:, 3] 
lmr_vel_x = lmr_data[:, 4] 

# -----------------------------------------------------------------------------
# Compute Equilibrium Values
# -----------------------------------------------------------------------------
eq = eqc.EqCalculator(['N2', 'O2', 'N', 'O', 'NO'])
X0 = eq.YtoX(np.array([0.802, 0.198, 0, 0, 0]))
s0 = GasState.from_pTv(p=133.3, T=300, v=4270, X0=X0, eq=eq)
s1 = normal_shock(s0)
Y1 = eq.XtoY(s1.X) 

# -----------------------------------------------------------------------------
# Temperature
# -----------------------------------------------------------------------------
plt.figure()

plt.plot([], [], color='k', linestyle='solid', label='Stratos')
plt.plot([], [], color='k', linestyle='dashed', label='Eilmer5')
plt.plot([], [], color='k', linestyle='dotted', label='eqc')

# T_tr
p_tr = plt.plot(strat['x'], strat['T_tr'], linestyle='solid')
c_tr = p_tr[0].get_color()
plt.plot([], [], color=c_tr, label=r"$T_{tr}$")
plt.plot(lmr_x, lmr_T_tr, color=c_tr, linestyle='dashed')

# T_ve
p_ve = plt.plot(strat['x'], strat['T_ve'], linestyle='solid')
c_ve = p_ve[0].get_color()
plt.plot([], [], color=c_ve, label=r"$T_{ve}$")
plt.plot(lmr_x, lmr_T_ve, color=c_ve, linestyle='dashed')

# Equilibrium T
plt.axhline(s1.T, color='gray', linestyle='dotted')

plt.xlabel('Distance behind shock, m')
plt.ylabel('Temperature, K')
plt.xlim(0, 0.2)
plt.legend()
plt.tight_layout()
plt.savefig('profile_T.png', dpi=300)

# -----------------------------------------------------------------------------
# Species (Mass Fractions)
# -----------------------------------------------------------------------------
plt.figure()

plt.plot([], [], color='k', linestyle='solid', label='Stratos')
plt.plot([], [], color='k', linestyle='dashed', label='Eilmer5')
plt.plot([], [], color='k', linestyle='dotted', label='eqc')

species = ['N2', 'O2', 'N', 'O', 'NO']
lmr_species_cols = {'N2': 9, 'O2': 10, 'N': 11, 'O': 12, 'NO': 13}

for i, sp in enumerate(species):
    p = plt.plot(strat['x'], strat['Y_' + sp], linestyle='solid')
    c = p[0].get_color()
    plt.plot([], [], color=c, label=sp)
    
    if sp in lmr_species_cols:
        col = lmr_species_cols[sp]
        plt.plot(lmr_x, lmr_data[:, col], color=c, linestyle='dashed')
        
    plt.axhline(Y1[i], color=c, linestyle='dotted')

plt.xlabel('Distance behind shock, m')
plt.ylabel('Mass fraction')
plt.xlim(0, 0.2)
plt.ylim(0, 0.85)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()
plt.savefig('profile_massf.png', dpi=300)

# -----------------------------------------------------------------------------
# Velocity
# -----------------------------------------------------------------------------
plt.figure()

plt.plot(strat['x'], strat['u'], label='Stratos')
plt.plot(lmr_x, lmr_vel_x, label='Eilmer5')
plt.axhline(s1.v, color='gray', linestyle='dotted', label='eqc')

plt.xlabel('Distance behind shock, m')
plt.ylabel('Velocity, m/s')
plt.xlim(0, 0.2)
plt.legend()
plt.tight_layout()
plt.savefig('profile_vel.png', dpi=300)

# -----------------------------------------------------------------------------
# Pressure
# -----------------------------------------------------------------------------
plt.figure()

plt.plot(strat['x'], strat['p'], label='Stratos')
plt.plot(lmr_x, lmr_p, label='Eilmer5')
plt.axhline(s1.p, color='gray', linestyle='dotted', label='eqc')

plt.xlabel('Distance behind shock, m')
plt.ylabel('Pressure, Pa')
plt.xlim(0, 0.2)
plt.legend()
plt.tight_layout()
plt.savefig('profile_p.png', dpi=300)

# -----------------------------------------------------------------------------
# Density
# -----------------------------------------------------------------------------
plt.figure()

plt.plot(strat['x'], strat['rho'], label='Stratos')
plt.plot(lmr_x, lmr_rho, label='Eilmer5')
plt.axhline(s1.rho, color='gray', linestyle='dotted', label='eqc')

plt.xlabel('Distance behind shock, m')
plt.ylabel(r'Density, kg/m$^3$')
plt.xlim(0, 0.2)
plt.legend()
plt.tight_layout()
plt.savefig('profile_rho.png', dpi=300)
