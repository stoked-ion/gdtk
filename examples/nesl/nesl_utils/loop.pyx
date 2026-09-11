# type: ignore
from time import time
import numpy as np

from libc.math cimport fabs, fmin, fmax, sqrt, INFINITY
from fast_gas cimport GasState, ThermochemicalReactor
from cpython.exc cimport PyErr_CheckSignals
cimport cython

include "velocity_gradient.pyx"
include "shock_curvature.pyx"
include "compute_dt.pyx"
include "flux.pyx"

cpdef object Loop(
    int MAX_ITERS, double T_END, double CFL_c, double CFL_d, int CELLS, double DOMAIN, double RADIUS, bint SENSOR, bint chemistry, double T_FROZEN, 
    bint VISCOSITY, str CATALYTIC, double[:] WALL_MASS_FRAC, double LEWIS_NUM, double[:] mol_masses, double T_WALL, bint VERBOSE, double[:] x_cen, 
    double[:] dx, double[:,:] dx2d, double[:] recom_eff, long[:] recom_products, int alpha, ThermochemicalReactor reactor, GasState[:] gas, int nsp,
    int nmodes, double rho_frozen, double[:,:] mass_frac_st, double[:,:] U, double[:] U_inflow, double[:,:] F_c, double[:,:] F_d, double[:,:] Q, 
    double[:] a_st, double[:] mu_st, double[:] rho_st, double[:] T_st, double[:] k_st, double[:] p_st, double[:] Cp_st, double[:] T_ve_st, double[:] k_ve_st,
    double residual, double[:] eq_wall_mass_frac, object transient_file, double transient_interval, bint LTS, str CURVATURE, bint BETA_SMOOTHING):
    """Main computational loop used in NESL."""

    # Solution Arrays.
    cdef double[:, :] mass_frac = np.empty_like(mass_frac_st, dtype=np.double)
    cdef double[:] r = np.abs(np.array(x_cen) - DOMAIN) + RADIUS
    cdef double[:] E_ve = np.zeros_like(rho_st, dtype=np.double)
    cdef double[:] e_ve = np.zeros_like(rho_st, dtype=np.double)  
    cdef double[:] rho = np.empty_like(rho_st, dtype=np.double)
    cdef double[:, :] U_old = np.empty_like(U, dtype=np.double)
    cdef double[:] dt = np.zeros_like(rho_st, dtype=np.double)
    cdef double[:] u = np.empty_like(rho_st, dtype=np.double)
    cdef double[:] E = np.empty_like(rho_st, dtype=np.double)
    cdef double[:] e = np.empty_like(rho_st, dtype=np.double)

    # Ghost Cell State.
    cdef double[:] mass_frac_ghost = np.empty(mass_frac_st.shape[1], dtype=np.double)
    cdef double T_local
    cdef GasState state

    # Source terms.
    cdef double[:] beta_st = np.zeros_like(rho_st, dtype=np.double)
    cdef double R_s, standoff, rho_2_avg, beta
    cdef int shock_idx

    # Index Shortcuts.
    cdef int idx_engy_ve = nsp + 2
    cdef int idx_engy = nsp + 1
    cdef int idx_xmom = nsp

    cdef int idx_two_from_wall = U.shape[0] - 4
    cdef int idx_one_from_wall = U.shape[0] - 3
    cdef int idx_wall_flux = F_d.shape[0] - 1
    cdef int idx_wall_ghost = U.shape[0] - 1
    cdef int idx_wall = U.shape[0] - 2

    cdef int U_rows = U.shape[0]
    cdef int U_cols = U.shape[1]

    # Loop Variables.
    cdef double rho_i, inv_rho_i, u_i, e_i, E_i, e_ve_i, E_ve_i
    cdef int i, j

    # Diffusion & Transport.
    cdef double mass_diffusion_isp, enthalpy_tr_isp, enthalpy_ve_isp, enthalpy_isp
    cdef double D_s, dT_dx, du_dx, dbeta_dx, k_w, Da_j, dx_cen_diff
    cdef double[:] mass_diffusion = np.zeros(nsp, dtype=np.double)
    cdef double mu_cen, k_cen, rho_cen, beta_cen, u_cen
    cdef double q_ve_x_conduction = 0.0
    cdef double q_ve_x_diffusion = 0.0
    cdef double q_x_conduction = 0.0
    cdef double q_x_diffusion = 0.0
    cdef double k_ve_cen = 0.0
    cdef double dT_ve_dx = 0.0

    # Shock Sensor.
    cdef double[:] S = np.zeros_like(rho_st, dtype=np.double)
    cdef double S_local = 1.0

    # Convergence & Time-Stepping.
    cdef double[:] dt_suggest = np.full(rho_st.shape[0], -1.0, dtype=np.double) 
    cdef double wall_clock_start = time()
    cdef double last_saved_time = -1.0 
    cdef double next_save = 0.0
    cdef double sim_time = 0.0
    cdef int steady_count = 0
    cdef double dU = 0.0
    cdef double diff_U
    cdef int iters = 0

    cdef double pi = np.pi

    while T_END < 0.0 or sim_time < T_END:
        # Check for user interrupt, ctrl+c.
        try:
            PyErr_CheckSignals()

        except KeyboardInterrupt:
            if VERBOSE:
                print("\nInterrupted by user, breaking loop and returning current state.")

            break

        # Apply Boundary Conditions to ghost cells.
        U[0, :] = U_inflow  # Inflow Conditions.

        # (Any or all of these may be overwritten in main loop).
        for i in range(nsp):  # Non-catalytic wall.
            U[idx_wall_ghost, i] = U[idx_wall, i]  

        U[idx_wall_ghost, idx_xmom] = -U[idx_wall, idx_xmom]  # Zero x-momentum.
        U[idx_wall_ghost, idx_engy] = U[idx_wall, idx_engy]  # Adiabatic wall.
        U[idx_wall_ghost, idx_engy_ve] = U[idx_wall, idx_engy_ve]  # Adiabatic wall.

        # Get Primitives.
        for i in range(CELLS + 2):
            rho_i = 0.0

            for j in range(nsp): 
                rho_i += U[i, j]
            
            inv_rho_i = 1.0 / rho_i
            u_i = U[i, idx_xmom] * inv_rho_i
            E_i = U[i, idx_engy] * inv_rho_i
            e_i = E_i - 0.5 * (u_i * u_i)
            E_ve_i = U[i, idx_engy_ve] * inv_rho_i
            e_ve_i = E_ve_i

            rho[i] = rho_i
            u[i] = u_i
            E[i] = E_i
            e[i] = e_i
            E_ve[i] = E_ve_i
            e_ve[i] = e_ve_i

            for j in range(nsp):
                mass_frac[i, j] = U[i, j] * inv_rho_i

        # Get stable timestep. This is missing vibro-electronic stability criterion, also wrong for 2T as uses Cp_bulk not Cp_tr.
        if LTS:
            compute_dt_lts(u, a_st, rho_st, mu_st, k_st, Cp_st, dt, dx, CFL_c, CFL_d, LEWIS_NUM)

        else:
            compute_dt(u, a_st, rho_st, mu_st, k_st, Cp_st, dt, dx, CFL_c, CFL_d, LEWIS_NUM)

        # Get shock radius.
        shock_curvature(u, x_cen, rho, dx, DOMAIN, RADIUS, CURVATURE, &R_s, &standoff, &shock_idx, alpha)
        
        # Solve for beta=dv/dy.
        if VISCOSITY: 
            beta_bvp(u, x_cen, rho_st, p_st, mu_st, DOMAIN, RADIUS, R_s, shock_idx, rho_frozen, beta_st)

            if BETA_SMOOTHING:
                smooth_beta(x_cen, p_st, beta_st, shock_idx) # Account for finite shock thickness.

        else:
            beta_ivp(u, x_cen, rho_st, p_st, DOMAIN, RADIUS, R_s, shock_idx, rho_frozen, beta_st)

        # Calculate fluxes and source terms.
        for i in range(1, CELLS + 2):
            # Update gas state.
            state = gas[i]
            state.rho = rho[i]
            state.u = e[i]
            state.massf[:] = mass_frac[i, :]
            state.T = T_st[i]  # Seed Newton solver for better convergence. 

            if nmodes == 1:
                state.u_modes[0] = e_ve[i]
                state.T_modes[0] = T_ve_st[i]  # Seed Newton solver for better convergence.

            state.push()

            if state.update_thermo_from_rhou() != 0:  # Abort cleanly if the gas update fails.
                raise RuntimeError(f"Gas update thermo_from_rhou failed in cell {i} at iter {iters}.")

            # Apply thermochemistry.
            if chemistry and state.T > T_FROZEN:
                dt_suggest[i] = reactor.update_state(state, dt[i], dt_suggest[i])

                if reactor.status != 0:  # Abort cleanly if the reactor fails.
                    raise RuntimeError(f"Thermochemical reactor failed in cell {i} at iter {iters}.")

            state.update_trans_coeffs()
            state.update_sound_speed()
            state.pull()

            # Update conserved variables (and other stored variables).
            for j in range(nsp):
                U[i, j] = state.massf[j] * state.rho

            U[i, idx_xmom] = state.rho * u[i]
            U[i, idx_engy] = state.rho * (state.u + 0.5 * u[i]**2)
            mass_frac_st[i] = state.massf
            T_st[i] = state.T
            p_st[i] = state.p
            mu_st[i] = state.mu
            k_st[i] = state.k
            a_st[i] = state.a
            rho_st[i] = state.rho
            Cp_st[i] = state.get_Cp()

            if nmodes == 1:
                U[i, idx_engy_ve] = state.rho * state.u_modes[0]
                k_ve_st[i] = state.k_modes[0]
                T_ve_st[i] = state.T_modes[0]

            # Assign beta=dv/dy term.
            beta = beta_st[i]

            # Geometric source terms.
            for j in range(nsp):
                Q[i, j] = -state.massf[j] * state.rho * beta * alpha

            Q[i, idx_xmom] = -state.rho * u[i] * beta * alpha
            Q[i, idx_engy] = -(state.rho * E[i] + state.p) * beta * alpha

            if nmodes == 1:
                Q[i, idx_engy_ve] = -(state.rho * E_ve[i]) * beta * alpha

            # Check and account for additional wall BC.
            if i > CELLS:
                if T_WALL > 0.0:  # One-sided difference applied later to avoid setting ghost cell state to -ve temperature.
                    state.T = T_WALL

                    if nmodes == 1:
                        state.T_modes[0] = T_WALL

                if CATALYTIC != "non_catalytic":
                    if CATALYTIC == "finite_rate":     
                        D_s = k_st[idx_wall] / (rho_st[idx_wall] * Cp_st[idx_wall] * LEWIS_NUM)  # In 2T implementation this should be Cp_tr, but, we can only get Cp_bulk from the GDTK. These are also all technically stale properties.

                        T_local = T_WALL if T_WALL > 0.0 else T_st[idx_wall]  # Assuming 2T is in equilibrium by now.

                        for j in range(nsp): 
                            if recom_eff[j] == 0.0:
                                mass_frac_ghost[j] = mass_frac_st[idx_wall, j]
                                continue

                            k_w = recom_eff[j] * sqrt(8.31451 * T_local / (2.0 * pi * mol_masses[j]))
                            Da_j = (k_w * dx[i]) / D_s
                            mass_frac_ghost[j] = (mass_frac_st[idx_wall, j] * (2.0 - Da_j) + 2.0 * Da_j * eq_wall_mass_frac[j]) / (2.0 + Da_j)

                        for j in range(nsp):
                            if recom_products[j] == -1:
                                continue
        
                            mass_frac_ghost[recom_products[j]] += mass_frac_st[idx_wall, j] - mass_frac_ghost[j] 

                    elif CATALYTIC == "fixed_composition":
                        for j in range(nsp):
                            mass_frac_ghost[j] = 2.0 * WALL_MASS_FRAC[j] - mass_frac_st[idx_wall, j]
                    
                    elif CATALYTIC == "equilibrium":
                        for j in range(nsp):
                            mass_frac_ghost[j] = 2.0 * eq_wall_mass_frac[j] - mass_frac_st[idx_wall, j]

                    state.massf[:] = mass_frac_ghost
                    mass_frac_st[idx_wall_ghost] = mass_frac_ghost
                
                state.push() # We do not state.update_trans_coeffs(), state.pull() here as T_ghost or Y_ghost can be -ve and will poison our transport properties.

            # Diffusive fluxes and source terms.
            if VISCOSITY: 
                # Jameson shock sensor.
                if i < CELLS and SENSOR:
                    S[i] = fabs((p_st[i + 1] - 2 * p_st[i] + p_st[i - 1]) / (p_st[i + 1] + 2 * p_st[i] + p_st[i - 1]))  # Reading stale p_st[i + 1], acts as artificial damping for steady but invalid for transient. 
                    S_local = 1 - fmin(1.0, 2.0 * fmax(fmax(S[i - 1], S[i]), S[i + 1])) 

                dx_cen_diff = x_cen[i] - x_cen[i - 1]  # Central difference at cell interface.
                mu_cen = S_local * (state.mu + mu_st[i - 1]) / 2.0
                k_cen = S_local * (state.k + k_st[i - 1]) / 2.0
                rho_cen = (state.rho + rho_st[i - 1]) / 2.0
                beta_cen = (beta + beta_st[i - 1]) / 2.0
                u_cen = (u[i] + u[i - 1]) / 2.0

                du_dx = (u[i] - u[i - 1]) / dx_cen_diff
                dT_dx = (state.T - T_st[i - 1]) / dx_cen_diff

                if i > CELLS and T_WALL > 0.0:
                    dT_dx = 2.0 * dT_dx 

                dbeta_dx = (beta -  beta_st[i - 1]) / dx_cen_diff  # Backward difference at cell centre.

                if nmodes == 1:
                    k_ve_cen = S_local * (state.k_modes[0] + k_ve_st[i - 1]) / 2.0
                    dT_ve_dx = (state.T_modes[0] - T_ve_st[i - 1]) / dx_cen_diff

                    if i > CELLS and T_WALL > 0.0:
                        dT_ve_dx = 2.0 * dT_ve_dx

                    q_ve_x_conduction = k_ve_cen * dT_ve_dx 

                q_x_conduction = k_cen * dT_dx 
                q_x_diffusion = 0.0

                D_s = k_cen / (rho_cen * Cp_st[i] * LEWIS_NUM)  # In 2T implementation this should be Cp_tr, but, we can only get Cp_bulk from the GDTK.

                for j in range(nsp):
                    mass_diffusion_isp = rho_cen * D_s * (state.massf[j] - mass_frac_st[i - 1, j]) / dx_cen_diff  # Fick's First Law. 
                    mass_diffusion[j] = mass_diffusion_isp
                    F_d[i - 1, j] = mass_diffusion_isp

                if nmodes == 1:
                    q_ve_x_diffusion = 0.0

                    for j in range(nsp):
                        enthalpy_isp = state.enthalpy_isp(j)
                        enthalpy_ve_isp = state.enthalpy_isp_in_mode(j, 0)
                        enthalpy_tr_isp = enthalpy_isp - enthalpy_ve_isp
                        q_x_diffusion += enthalpy_tr_isp * mass_diffusion[j]  
                        q_ve_x_diffusion += enthalpy_ve_isp * mass_diffusion[j] 

                else:
                    for j in range(nsp):
                        q_x_diffusion += state.enthalpy_isp(j) * mass_diffusion[j]  

                F_d[i - 1, idx_xmom] = 2.0 / 3.0 * mu_cen * (2.0 * du_dx - beta_cen * alpha)
                F_d[i - 1, idx_engy] = 2.0 / 3.0 * mu_cen * (2.0 * du_dx - beta_cen * alpha) * u_cen + q_x_conduction + q_x_diffusion
                F_d[i - 1, idx_engy_ve] = q_ve_x_conduction + q_ve_x_diffusion

                Q[i, idx_xmom] += state.mu * dbeta_dx * alpha
                Q[i, idx_engy] += state.mu * (dbeta_dx * u[i] + 2.0 / 3.0 * beta * (2 * beta - du_dx)) * alpha - (state.k * dT_dx / r[i]) * alpha

                if nmodes == 1:
                    Q[i, idx_engy_ve] -= (state.k_modes[0] * dT_ve_dx / r[i]) * alpha

            # Convective fluxes.
            hllc(u[i - 1], rho_st[i - 1], p_st[i - 1], e[i - 1], e_ve[i - 1], a_st[i - 1], mass_frac_st[i - 1],
                 u[i], state.rho, state.p, e[i], e_ve[i], state.a, state.massf, nsp, nmodes, F_c[i - 1, :]) 

        # Enforce correct convective wall flux, only pressure normal to wall as u=0 on the wall (no contact speed).
        F_c[idx_wall_flux, :] = 0.0
        F_c[idx_wall_flux, idx_xmom] = (15.0 * p_st[idx_wall] - 10.0 * p_st[idx_one_from_wall] + 3.0 * p_st[idx_two_from_wall]) / 8.0

        # Update conserved variables.
        for i in range(1, U_rows - 1):
            for j in range(U_cols):
                U[i, j] += -(dt[i] / dx2d[i, j]) * ((F_c[i, j] - F_d[i, j]) - (F_c[i - 1, j] - F_d[i - 1, j])) + (Q[i, j] * dt[i])

        # Check for steady state.
        if iters > 0:
            dU = 0.0 

            for i in range(U_rows):
                for j in range(U_cols):
                    diff_U = U[i, j] - U_old[i, j]
                    
                    if diff_U < 0.0:
                        diff_U = -diff_U
                    
                    if diff_U > dU:
                        dU = diff_U

            if T_END < 0.0:
                if dU < residual:
                    steady_count += 1

                else:
                    steady_count = 0

                if steady_count > 100:
                    if VERBOSE:
                        print("Steady state reached.")
                        
                    break

        U_old[:, :] = U[:, :]
        
        if VERBOSE:
            if iters == 0:
                if LTS:
                    print(f"Iters : {iters}, Wall Clock : {(time() - wall_clock_start):.2f}")

                else:
                    print(f"Iters : {iters}, Sim Time : {sim_time:.2e}, Wall Clock : {(time() - wall_clock_start):.2f}")

            if iters % 1000 == 0 and iters > 0:
                if LTS:
                    print(f"Iters : {iters}, Wall Clock : {(time() - wall_clock_start):.2f}, Residual : {dU:.2e}")

                else:
                    print(f"Iters : {iters}, Sim Time : {sim_time:.2e}, Wall Clock : {(time() - wall_clock_start):.2f}, Residual : {dU:.2e}") 

        # Transient stagnation point trace.
        if transient_file is not None and sim_time >= next_save:
            transient_file.write(
                f"{sim_time:.10e}, {standoff:.10e}, "
                f"{q_x_conduction + q_ve_x_conduction:.10e}, "
                f"{q_x_diffusion + q_ve_x_diffusion:.10e}, {p_st[idx_wall]:.10e}\n")
            next_save += transient_interval
            last_saved_time = sim_time

        # Utils.
        sim_time += dt[1]
        iters += 1

        if iters >= MAX_ITERS:
            if LTS:
                print(f"Iters : {iters}, Wall Clock : {(time() - wall_clock_start):.2f}, Residual : {dU:.2e}")

            else:
                print(f"Iters : {iters}, Sim Time : {sim_time:.2e}, Wall Clock : {(time() - wall_clock_start):.2f}, Residual : {dU:.2e}")

            print("Maximum iterations completed.")
            break

    # Guaranteed final timestep stagnation point trace.
    if transient_file is not None and iters > 0 and sim_time != last_saved_time:
        transient_file.write(
            f"{sim_time:.10e}, {standoff:.10e}, "
            f"{q_x_conduction + q_ve_x_conduction:.10e}, "
            f"{q_x_diffusion + q_ve_x_diffusion:.10e}, {p_st[idx_wall]:.10e}\n")

    return x_cen, u, rho_st, T_st, T_ve_st, p_st, e, e_ve, mu_st, k_st, k_ve_st, a_st, beta_st, mass_frac_st, [q_x_conduction + q_ve_x_conduction, q_x_diffusion + q_ve_x_diffusion], R_s