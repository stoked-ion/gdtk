import matplotlib.pyplot as plt
from stratos import Stratos
import numpy as np

# -----------------------------------------------------------------------------
# Simulation Setup
# -----------------------------------------------------------------------------
st = Stratos('air-7sp.gas', 'air-7sp.chem', '', 1.0, 100000, 4273.64, {"N2":0.78, "O2":0.22}, 300, 133.3)
strat = st.normal()

# -----------------------------------------------------------------------------
# Load Reference Data
# -----------------------------------------------------------------------------
marrone_T = np.loadtxt('ref_data/marrone_fig4_T_ratio.g3data')
marrone_rho = np.loadtxt('ref_data/marrone_fig4_rho_ratio.g3data')
poshax = np.loadtxt('ref_data/poshax.data')

marrone_N2 = np.loadtxt('ref_data/marrone_fig3_N2.g3data')
marrone_O2 = np.loadtxt('ref_data/marrone_fig3_O2.g3data')
marrone_NO = np.loadtxt('ref_data/marrone_fig3_NO.g3data')
marrone_O = np.loadtxt('ref_data/marrone_fig3_O.g3data')
marrone_N = np.loadtxt('ref_data/marrone_fig3_N.g3data')
marrone_NO_e = np.loadtxt('ref_data/marrone_fig3_NO+_e.g3data')

# -----------------------------------------------------------------------------
# Plotting Setup
# -----------------------------------------------------------------------------
species = ['N2', 'O2', 'NO', 'O', 'N', 'NO+']
marrone_data = [marrone_N2, marrone_O2, marrone_NO, marrone_O, marrone_N, marrone_NO_e]
masses = [0.0280134, 0.0319988, 0.0300061, 0.0159994, 0.0140067, 0.0300055514]

poshax_idx = [7, 9, 15, 13, 11, 17]
poshax_mass_idx = [6, 8, 14, 12, 10, 16]

mole_norm = 4.092e-01 # Normalisation factor for concentration

# -----------------------------------------------------------------------------
# Temperature and Density Ratios
# -----------------------------------------------------------------------------
plt.figure()

plt.plot([], [], color='k', linestyle='solid', label='Stratos')
plt.plot([], [], color='k', linestyle='dashed', label='Marrone')
plt.plot([], [], color='k', linestyle='dotted', label='Poshax')

plt.plot([], [], color='blue', linestyle='solid', label=r'$T/T_\infty$')
plt.plot([], [], color='red', linestyle='solid', label=r'$\rho/\rho_\infty$')

# Temperature 
plt.plot(strat['x'] * 100, strat['T_tr'] / 300.0, color='blue', linestyle='solid')
plt.plot(marrone_T[:, 0], marrone_T[:, 1], color='blue', linestyle='dashed')
plt.plot(poshax[:, 0] * 100, poshax[:, 1] / 300.0, color='blue', linestyle='dotted')

# Density 
plt.plot(strat['x'] * 100, strat['rho'] / 0.00153923, color='red', linestyle='solid')
plt.plot(marrone_rho[:, 0], marrone_rho[:, 1], color='red', linestyle='dashed')
plt.plot(poshax[:, 0] * 100, poshax[:, 3] / 0.00153923, color='red', linestyle='dotted')

plt.xlabel("Distance behind shock, cm")
plt.ylabel("Ratio")
plt.xscale('log')
plt.xlim(0.01, 100)
plt.legend()
plt.tight_layout()
plt.savefig('profile_T_rho.png', dpi=300)

# -----------------------------------------------------------------------------
# Species (Moles per original mole)
# -----------------------------------------------------------------------------
plt.figure()

plt.plot([], [], color='k', linestyle='solid', label='Stratos')
plt.plot([], [], color='k', linestyle='dashed', label='Marrone')
plt.plot([], [], color='k', linestyle='dotted', label='Poshax')

for sp, mass, p_idx, p_midx, m_data in zip(species, masses, poshax_idx, poshax_mass_idx, marrone_data):
    p = plt.plot(strat['x'] * 100, strat['Y_' + sp] * strat['rho'] / (mole_norm * mass), linestyle='solid')
    c = p[0].get_color()
    plt.plot([], [], color=c, label=sp)
    
    plt.plot(m_data[:, 0], m_data[:, 1], color=c, linestyle='dashed')
    plt.plot(poshax[:, 0] * 100, poshax[:, p_idx] / mole_norm, color=c, linestyle='dotted')

plt.xlabel("Distance behind shock, cm")
plt.ylabel("Moles per original mole")
plt.xscale('log') 
plt.yscale('log')
plt.xlim(0.01, 100)
plt.ylim(1e-7, 2)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()
plt.savefig('profile_moles.png', dpi=300)

# -----------------------------------------------------------------------------
# Species (Mass Fractions)
# -----------------------------------------------------------------------------
plt.figure()

plt.plot([], [], color='k', linestyle='solid', label='Stratos')
plt.plot([], [], color='k', linestyle='dotted', label='Poshax')

for sp, p_midx in zip(species, poshax_mass_idx):
    p = plt.plot(strat['x'] * 100, strat['Y_' + sp], linestyle='solid')
    c = p[0].get_color()
    plt.plot([], [], color=c, label=sp)
    
    plt.plot(poshax[:, 0] * 100, poshax[:, p_midx], color=c, linestyle='dotted')

plt.xlabel("Distance behind shock, cm")
plt.ylabel("Mass fraction")
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.01, 100)
plt.ylim(1e-7, 2)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()
plt.savefig('profile_massf.png', dpi=300)
