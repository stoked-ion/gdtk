#!/bin/sh
"exec" "$DGD/lib/.nesl/bin/python3" "-B" "$0" "$@"

"""
A program that simulates the stagnation-line of blunt 
bodies. Relies on a quasi-1D formulation with source 
terms calculated from the transverse velocity gradient. 

Author: Taine J. Rossini (t.rossini@uq.edu.au)
Build Date: July 29, 2026

"""

__version__ = "1.0.0"

# pyright: reportUndefinedVariable=false, reportMissingImports=false
from normalshock import GasState as eqcGasState, normal_shock as eqc_normal_shock
from nesl_utils.fast_gas import GasState, GasFlow 
from nesl_utils import Helper, Grid, Prep, Plot
from os.path import dirname, join, splitext
from nesl_utils.loop import Loop
from sys import path
import numpy as np
import eqc

def main() -> None:
    """Initialise the simulation and handle the prep."""
    Prep.toml(globals(), __version__)

    gmodel = Prep.gas(GAS_MODEL)

    if chemistry:
        reactor = Prep.chem(CHEM_MODEL, EXCHANGE_MODEL, gmodel)

    else:
        reactor = None

    if VERBOSE: print("Initialising...")

    # Setup gas and flow models.
    flow = GasFlow(gmodel)
    nsp = gmodel.numSpecies
    nmodes = gmodel.numModes
    state_inf = GasState(gmodel)
    state2 = GasState(gmodel)
    gas = np.array([GasState(gmodel) for _ in range(CELLS + 2)])
    mol_masses = gmodel.molMasses

    # Assign freestream values.
    state_inf.T = T_INF
    state_inf.massf = MASS_FRAC
    state_inf.T_modes = np.array([T_INF])

    if P_INF is not None:
        state_inf.p = P_INF
        state_inf.push()
        state_inf.update_thermo_from_pT()

    else:
        state_inf.rho = RHO_INF
        state_inf.push()
        state_inf.update_thermo_from_rhoT()
        
    state_inf.update_sound_speed()
    state_inf.update_trans_coeffs()
    state_inf.pull()

    for i in range(CELLS + 2):
        gas[i].copy_values(state_inf._stateId)
        gas[i].pull()

    # eqc at stagnation point.
    eq_wall_mass_frac = np.array([])
    
    if chemistry:
        eq = eqc.EqCalculator(gmodel.nameSpecies)
        X0 = eq.YtoX(MASS_FRAC)
        s0 = eqcGasState.from_pTv(p=state_inf.p, T=T_INF, v=U_INF, X0=X0, eq=eq)
        s1 = eqc_normal_shock(s0)

        if CATALYTIC != "non_catalytic":
            p_stag = s0.pitot_pressure()
            T_wall = s1.T if T_WALL < 0.0 else T_WALL
            X1 = eq.pt(p=p_stag, T=T_wall, Xs0=X0)
            eq_wall_mass_frac = eq.XtoY(X1)

    # Define conserved variables: [rho_i, ..., rho*u, rho*E_tr, rho*E_ve].
    U = np.zeros((CELLS + 2, nsp + 2 + 1), dtype=np.double)

    # Define arrays.
    F_c = np.zeros((CELLS + 1, nsp + 2 + 1), dtype=np.double)
    F_d = np.zeros((CELLS + 1, nsp + 2 + 1), dtype=np.double)
    Q = np.zeros((CELLS + 2, nsp + 2 + 1), dtype=np.double)
    T_st = np.full(CELLS + 2, T_INF, dtype=np.double)
    p_st = np.full(CELLS + 2, state_inf.p, dtype=np.double)
    mu_st = np.full(CELLS + 2, state_inf.mu, dtype=np.double)
    k_st = np.full(CELLS + 2, state_inf.k, dtype=np.double)
    a_st = np.full(CELLS + 2, state_inf.a, dtype=np.double)
    Cp_st = np.full(CELLS + 2, state_inf.get_Cp(), dtype=np.double)
    rho_st = np.full(CELLS + 2, state_inf.rho, dtype=np.double)
    mass_frac_st = np.full((CELLS + 2, nsp), state_inf.massf, dtype=np.double)

    if nmodes == 1:
        k_ve_st = np.full(CELLS + 2, state_inf.k_modes[0], dtype=np.double)
        T_ve_st = np.full(CELLS + 2, state_inf.T_modes[0], dtype=np.double)

    else:
        k_ve_st = np.zeros(CELLS + 2, dtype=np.double)
        T_ve_st = np.zeros(CELLS + 2, dtype=np.double)

    # Grid generation.
    x_cen, dx, dx2d = Grid(CELLS, DOMAIN, CLUSTERING, nsp).make()

    # Initial flow conditions.
    U[:, :nsp] = np.array(state_inf.massf, dtype=np.double) * state_inf.rho
    U[:, nsp] = state_inf.rho * U_INF
    U[:, nsp + 1] = state_inf.rho * (state_inf.u + 0.5 * U_INF**2)

    if nmodes == 1:
        U[:, nsp + 2] = state_inf.rho * state_inf.u_modes[0]

    else:
        U[:, nsp + 2] = 0.0

    U_inflow = U[0, :].copy()

    # Hot start simulation.  
    v2, _ = flow.normal_shock(state_inf, U_INF, state2)

    if chemistry:
        epsilon = state_inf.rho / s1.rho

    else:
        epsilon = state_inf.rho / state2.rho

    delta = Helper.shock_standoff(epsilon, RADIUS, AXISYMMETRIC)
    idx = Helper.find_idx(x_cen, (DOMAIN - delta))

    if HOT_START:
        U[idx:, :nsp] = np.array(state_inf.massf, dtype=np.double) * state2.rho
        U[idx:, nsp] = state2.rho * np.linspace(v2, 0, len(U[idx:, nsp]))
        U[idx:, nsp + 1] = state2.rho * (state2.u + 0.5 * (U[idx:, nsp] / state2.rho)**2)

        if nmodes == 1:
            U[idx:, nsp + 2] = state2.rho * state2.u_modes[0]

    if VERBOSE: print("Beginning to step through time...")

    transient_file = None
    if transient_interval >= 0.0: 
        transient_file = open("transient_" + SAVE, "w")
        transient_file.write("t (s), standoff (m), q_c (W/m^2), q_d (W/m^2), p (Pa)\n")

    # Pass off intensive loop to compiled Cython.
    try:
        x_cen, u, rho_st, T_st, T_ve_st, p_st, e, e_ve, mu_st, k_st, k_ve_st, a_st, beta_st, mass_frac_st, q_x, R_s = Loop(
            MAX_ITERS, T_END, CFL_c, CFL_d, CELLS, DOMAIN, RADIUS, SENSOR, chemistry, T_FROZEN, VISCOSITY, CATALYTIC, WALL_MASS_FRAC,
            LEWIS_NUM, mol_masses, T_WALL, VERBOSE, x_cen, dx, dx2d, recom_eff, recom_products, alpha, reactor, gas, nsp, nmodes, 
            state2.rho, mass_frac_st, U, U_inflow, F_c, F_d, Q, a_st, mu_st, rho_st, T_st, k_st, p_st, Cp_st,T_ve_st, k_ve_st, 
            residual, eq_wall_mass_frac, transient_file, transient_interval, LTS, CURVATURE, BETA_SMOOTHING)

    except RuntimeError as err:
        print(f"\nSimulation aborted: {err}")
        raise SystemExit(1)

    finally:
        if transient_file is not None:
            transient_file.close()

    # Calculate shock standoff.
    shock_idx = np.argmax(np.abs(np.diff(u) / np.diff(x_cen)))
    delta = DOMAIN - x_cen[shock_idx]

    if VERBOSE:
        print("Simulation complete.")

        # Print stagnation point heat flux.
        if VISCOSITY:
            print(f"Stagnation point convective heat flux is {q_x[0]:.2f} (W/m^2).")

            if CATALYTIC != "non_catalytic":
                print(f"Stagnation point diffusive heat flux is {q_x[1]:.2f} (W/m^2).")
                print(f"Stagnation point total heat flux is {np.sum(q_x):.2f} (W/m^2).")

        print(f"Shock standoff is {delta:.8f} (m).")

    # Save Data.
    if SAVE is not False:    
        if transient_file is None:
            np.savetxt("stagpoint_" + SAVE, np.column_stack([delta, q_x[0], q_x[1], p_st[-2], R_s]), delimiter=", ", header="standoff (m), q_c (W/m^2), q_d (W/m^2), p (Pa), R_s (m)", comments="")

        mass_frac_list = [list(mf) for mf in mass_frac_st[1:-1]]

        if nmodes == 1:
            data = np.column_stack((x_cen[1:-1], u[1:-1], rho_st[1:-1], T_st[1:-1], p_st[1:-1], e[1:-1], e_ve[1:-1], mu_st[1:-1], k_st[1:-1], a_st[1:-1], beta_st[1:-1], T_ve_st[1:-1], k_ve_st[1:-1]))
            data = np.hstack([data, np.array(mass_frac_list)])
            np.savetxt(
                SAVE,
                data,
                delimiter=", ",
                header="x (m), u (m/s), rho (kg/m^3), T_tr (K), p (Pa), e_tr (J/kg), e_ve (J/kg), mu (Pa·s), k_tr (W/(m·K)), a (m/s), beta (1/s), T_ve (K), k_ve (W/(m·K)), " + ", ".join(gmodel.nameSpecies),
                comments=""
            )

        else:
            data = np.column_stack((x_cen[1:-1], u[1:-1], rho_st[1:-1], T_st[1:-1], p_st[1:-1], e[1:-1], mu_st[1:-1], k_st[1:-1], a_st[1:-1], beta_st[1:-1]))
            data = np.hstack([data, np.array(mass_frac_list)])
            np.savetxt(
                SAVE,
                data,
                delimiter=", ",
                header="x (m), u (m/s), rho (kg/m^3), T (K), p (Pa), e (J/kg), mu (Pa·s), k (W/(m·K)), a (m/s), beta (1/s), " + ", ".join(gmodel.nameSpecies),
                comments=""
            )

    # Plot data.
    if PLOT and SAVE is not False:
        name, _ = splitext(SAVE)

        if nmodes == 1:
            Plot.prim_two_T(SAVE, name + "_prim")

        else:
            Plot.prim(SAVE, name + "_prim")
        
        if chemistry:
            Plot.massf(SAVE, name + "_massf", gmodel.nameSpecies)

        if transient_file is not None:
            Plot.transient("transient_" + SAVE, name + "_transient")

if __name__ == "__main__":
    BASE = dirname(dirname(__file__))
    path.insert(0, join(BASE, "lib"))
    
    main()