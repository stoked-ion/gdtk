from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from normalshock import GasState
import matplotlib.pyplot as plt
from stratos import Stratos
import numpy as np
import eqc

# -----------------------------------------------------------------------------
# Simulation Setup
# -----------------------------------------------------------------------------
T1 = 1480.9; u1 = 4007.41; p1 = 17517.90; rho1 = 0.0392844
massf1 = {"N2":0.867488, "O2":0.0572876, "N":0.0000000118, "O":0.0326201, "NO":0.0426042}

cells = 1000; nodes = 50; t_end = 2e-7; x_end = 0.01

st = Stratos('air-5sp.gas', 'air-5sp.chem', '', x_end, cells, u1, massf1, T1, rho1=rho1)

mach_lim = 0.8
extrap_mode = 'mach'  # 'mach' or 'R_p'.

# -----------------------------------------------------------------------------
# Load Reference Data
# -----------------------------------------------------------------------------
shock_shape = np.loadtxt('shock_shape.csv', delimiter=',')
body_shape = np.loadtxt('body_shape.csv', delimiter=',')
lmr = np.loadtxt('cfd.csv', delimiter=',')
ideal = np.loadtxt('ideal.csv', delimiter=',')
cea = np.loadtxt('cea.csv', delimiter=',')

# Shock.
spline = UnivariateSpline(shock_shape[:, 1], shock_shape[:, 0], s=0.0001)
spline_dash = spline.derivative(n=1)

y = np.linspace(shock_shape[:, 1].min(), shock_shape[:, 1].max(), nodes)
x = spline(y)

# Body.
spline_body = UnivariateSpline(body_shape[:, 1], body_shape[:, 0], s=0.0)
spline_body_dash = spline_body.derivative(n=1)

# -----------------------------------------------------------------------------
# Get Equilibrium properties
# -----------------------------------------------------------------------------
eq = eqc.EqCalculator(list(massf1.keys()))
X1 = eq.YtoX(np.array(list(massf1.values())))
s1 = GasState.from_pTv(p=p1, T=T1, v=u1, X0=X1, eq=eq)
p_stag = s1.pitot_pressure()

p_newtonian = lambda theta: p1 + (p_stag - p1) * np.sin(theta)**2 

# -----------------------------------------------------------------------------
# Loop through Nodes
# -----------------------------------------------------------------------------
density_ratio = np.empty_like(x)
density_ratio_newt = np.empty_like(x)
density_ratio_extrap = np.empty_like(x)

extrapolating = False
extrap_gradient = 0.0
extrap_last_val = None

last_y = None
y_m1 = None

intersection_eq = lambda yb: yb - m * spline_body(yb)

for i in range(nodes):
    beta = np.arctan(1.0 / spline_dash(y[i]))  # type: ignore

    columns = st.shock_angle(beta)
    st.interpolate_to_length(cells, t_end, columns, 't')

    m = y[i] / x[i]
    y_int = brentq(intersection_eq, a=body_shape[:, 1].min(), b=body_shape[:, 1].max())

    p = columns['p'][-1]
    R = columns['R'][-1]
    T = columns['T_tr'][-1]
    rho = columns['rho'][-1]
    gamma = columns['gamma'][-1]

    theta = abs(np.arctan(1.0 / spline_body_dash(y_int)))  # type: ignore

    mach = columns['u'][-1] / np.sqrt(gamma * R * T)

    density_ratio[i] = rho / rho1  # Does not account for stagnating / expanding on or around the body, relies on oblique shock relations.
    
    R_p_pure = (p_newtonian(theta) / p)**((gamma - 1) / gamma)  # Note this is polytropic not isentropic.
    pure_rho_ratio = rho / rho1 * R_p_pure
    density_ratio_newt[i] = pure_rho_ratio
    
    # Begin Extrapolation.
    pure_val = R_p_pure if extrap_mode == 'R_p' else pure_rho_ratio
    
    if extrapolating:
        current_val = extrap_last_val + extrap_gradient * (y[i] - last_y)

    else:
        current_val = pure_val
        if mach > mach_lim and extrap_last_val is not None:
            extrapolating = True
            extrap_gradient = (current_val - extrap_last_val) / (y[i] - last_y)

    extrap_last_val = current_val
    last_y = y[i]

    if extrap_mode == 'R_p':
        density_ratio_extrap[i] = rho / rho1 * current_val

    else:
        density_ratio_extrap[i] = current_val

    if mach > mach_lim and y_m1 is None:
        y_m1 = y[i]

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
plt.figure(figsize=(8,8))
plt.plot(density_ratio, y, label='Stratos')
plt.plot(density_ratio_newt, y, label='Stratos + Newt')

if extrap_mode == 'mach':
    plt.plot(density_ratio_extrap, y, label='Stratos + Newt + Mach Extrap')

elif extrap_mode == 'R_p':
    plt.plot(density_ratio_extrap, y, label='Stratos + Newt + R_p Extrap')

plt.plot(lmr[:, 0], lmr[:, 1], label='James lmr')
plt.plot(cea[:, 0], cea[:, 1], label='James cea')
plt.plot(ideal[:, 0], ideal[:, 1], label='James ideal')
plt.axhline(y_m1, linestyle='--', label=f'Mach Limit = {mach_lim}')  # type: ignore

plt.xlim(3.5, 8.5)
plt.ylim(0.0, 0.0117)
plt.xlabel(r'$\rho_2 / \rho_1$')
plt.ylabel('y, m')
plt.tight_layout()
plt.legend()
plt.savefig('density_ratio.png', dpi=300)
