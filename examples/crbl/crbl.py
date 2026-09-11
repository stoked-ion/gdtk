"""
A program to simulate chemically reacting boundary-layers.
Built on a space marching FDM scheme solving the steady 2D 
boundary-layer equations using the Levy-Lees transformation.

Author: Taine Rossini (t.rossini@uq.edu.au)
Build Date: May 7, 2026

"""

__version__ = "1.0.0"

from gdtk.gas import GasFlow, GasModel, GasState, ThermochemicalReactor
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d
from scipy.linalg import solve_banded
from scipy.optimize import curve_fit
from os.path import exists
from time import time
import pyvista as pv
from sys import argv
import numpy as np
import eqc

class GlobalConfig():
    """Contains parameters for solver configuration."""
    __slots__ = ('eta_end', 'n_y_cells', 'n_x_cells', 'gas_model_file',
                'chem_model_file', 'reacting', 'max_iters', 'tol', 
                'omega', 'axisymmetric', 'gmodel', 'reactor',
                'verbose', 'adiabatic', 'catalytic', 'export')

    def __init__(self) -> None:
        # Simulation parameters.
        self.gas_model_file = ""
        self.chem_model_file = ""
        self.reacting = False
        self.axisymmetric = False
        self.adiabatic = False
        self.catalytic = False
        self.verbose = True
        self.export = True

        # Grid parameters.
        self.eta_end: float = 6.0  # End of eta domain, where freestream begins.
        self.n_y_cells: int = 100  
        self.n_x_cells: int = 100  
        
        # Picard iteration parameters.
        self.max_iters: int = 100
        self.tol: float = 1e-4
        self.omega: float = 0.1 # Under-relaxation factor .

        # Initialise gas model variables.
        self.gmodel = None
        self.reactor = None
    
    def init_gas_model(self) -> None:
        """Initialise a GasModel object for use in the user's input script."""
        if self.gmodel:
            raise RuntimeError("Already have a gas model initialised.")
        
        if not exists(self.gas_model_file):
            raise Exception("Gas model file not found: " + self.gas_model_file)
        
        self.gmodel = GasModel(self.gas_model_file)

        if self.chem_model_file:
            if not exists(self.chem_model_file):
                raise Exception("Chemistry model file not found: " + self.chem_model_file)
            
            reactor = ThermochemicalReactor(self.gmodel, self.chem_model_file)

        else:
            reactor = None

        self.reactor = reactor

class BoundaryLayer():
    """Solves the boundary layer flow over an arbritrary geometry."""
    def __init__(self, x_arr, u_e_arr, T_e_arr, P_e_arr, massf_e_arr, r_arr=None, T_wall=300.0) -> None:
        # Some general setup.
        self.start = time()

        global config
        global active_bl

        active_bl = self
        config.n_x_cells = len(x_arr) 

        # User defined flow parameters as a function of the body arc length x.
        self.massf_e_arr = massf_e_arr
        self.u_e_arr = u_e_arr
        self.P_e_arr = P_e_arr
        self.T_e_arr = T_e_arr
        self.r_arr = r_arr
        self.T_wall = T_wall

        # Grid setup.
        self.x_arr = x_arr
        self.eta = np.linspace(0, config.eta_end, config.n_y_cells)
        self.delta_eta = self.eta[1] - self.eta[0]

        # Initialise gas objects.
        self.gmodel_type = config.gmodel.type_str
        self.working_state = GasState(config.gmodel)
        self.eq = eqc.EqCalculator(config.gmodel.species_names)

        # Additional global parameters.
        if config.axisymmetric and self.r_arr is None:
            print("\n'r_arr' was not specified. Continuing with planar simulation.")
            config.axisymmetric = False

        if config.axisymmetric:
            self.k = 1.0 
            
        else:
            self.k = 0.0
            self.r_arr =  np.ones(config.n_x_cells) 

        if config.adiabatic:  # Add check as adiabtic wall with chemistry is currently not working.
            if config.verbose:
                print("\nChemistry has been disabled, not implemented for reacting flows.")
                
            config.reacting = False
            config.catalytic = False
        
        # Check solver will capture physics.
        if config.reacting and config.verbose:
            if config.tol > 1e-5:
                print("\nDecreasing 'config.tol' suggested for reacting flows.")

        # Initialise additional edge parameters.
        self.rho_e_arr = np.empty(config.n_x_cells)
        self.mu_e_arr = np.empty(config.n_x_cells)
        self.h_e_arr = np.empty(config.n_x_cells)
        self.dp_e_dx_arr = np.empty(config.n_x_cells)
        self.xi_arr = np.empty(config.n_x_cells)
        self.delta_xi_arr = np.empty(config.n_x_cells)
        self.H_wall_arr = np.empty(config.n_x_cells)
        self.calculate_edge_arrays() 
               
    def calculate_edge_arrays(self) -> None:
        """Builds edge arrays globally."""
        integrand = np.empty(config.n_x_cells)
        for i in range(config.n_x_cells):
            # Edge properties.
            self.working_state.p = self.P_e_arr[i]
            self.working_state.T = self.T_e_arr[i]
            self.working_state.massf =  self.massf_e_arr[i].tolist()

            self.working_state.update_thermo_from_pT()
            self.working_state.update_trans_coeffs()
            
            self.rho_e_arr[i] = self.working_state.rho
            self.mu_e_arr[i] = self.working_state.mu
            self.h_e_arr[i] = config.gmodel.enthalpy(self.working_state)
            
            integrand[i] = self.u_e_arr[i] * self.rho_e_arr[i] * self.mu_e_arr[i] * self.r_arr[i]**(2 * self.k)

            # Wall properties.
            self.working_state.T = self.T_wall
            self.working_state.update_thermo_from_pT()

            self.H_wall_arr[i] = config.gmodel.enthalpy(self.working_state)

        self.dp_e_dx_arr = np.gradient(self.P_e_arr, self.x_arr)

        tmp_x_arr = np.insert(self.x_arr, 0, 0.0)
        integrand = np.insert(integrand, 0, integrand[0])
        self.xi_arr = cumulative_trapezoid(integrand, tmp_x_arr)

        self.delta_xi_arr = np.insert(np.diff(self.xi_arr), 0, 1.0)

    def get_T(self, h_target, T_guess_prev, tol=0.1, max_iter=20) -> float:
        """Newton solver using the previous iteration's temperature as the guess"""
        T_guess = T_guess_prev
        for i in range(max_iter): 
            self.working_state.T = T_guess
            self.working_state.update_thermo_from_pT()

            h_current = config.gmodel.enthalpy(self.working_state)
            Cp_current = config.gmodel.Cp(self.working_state)

            dh = h_current - h_target

            if abs(dh) < tol:
                break

            T_guess = T_guess - (dh / Cp_current)
            
        return T_guess

    def get_EQ(self, P, T, massf) -> np.ndarray:
        """Calculate equilibrium gas state."""
        X0 = self.eq.YtoX(massf)
        X1 = self.eq.pt(p=P, T=T, Xs0=X0)
        return self.eq.XtoY(X1)

    def solve_momentum(self, U_old, U_i_m_one, f, f_i_m_one, C, beta, rho_e_on_rho, xi, delta_xi) -> np.ndarray:
        """Solves (C U')' + f U' + beta(rho_e/rho - U^2) = 2xi(U' dU'/dxi - U'' df/dxi)"""
        N = config.n_y_cells
        
        # Calculate gradient of C.
        C_dash = np.gradient(C, self.delta_eta)
        
        # Create arrays.
        A = np.zeros(N)
        B = np.zeros(N)
        C_diag = np.zeros(N)  # Call C_diag to avoid conflict with C for Chapmin-Rubesin constant.
        D = np.zeros(N)
        
        # Matrix coeffeicients to solve TDMA (a_j * U_i,j-1) + (b_j * U_i,j) + (c_j * U_i,j+1) = d_j.
        # P,Q,R,S have been substituted in directly to avoid creation of additional arrays.
        A[1:-1] = C[1:-1] / (self.delta_eta**2) - (C_dash[1:-1] + f[1:-1] + 2 * xi / delta_xi  * (f[1:-1] - f_i_m_one[1:-1])) / (2 * self.delta_eta)
        B[1:-1] = -2 * C[1:-1] / (self.delta_eta**2) - beta * U_old[1:-1] - 2 * xi / delta_xi * U_old[1:-1]
        C_diag[1:-1] = C[1:-1] / (self.delta_eta**2) + (C_dash[1:-1] + f[1:-1] + 2 * xi / delta_xi  * (f[1:-1] - f_i_m_one[1:-1])) / (2 * self.delta_eta)
        D[1:-1] = -beta * rho_e_on_rho[1:-1] - 2 * xi / delta_xi * U_old[1:-1] * U_i_m_one[1:-1]
        
        # Boundary conditions (U=0 at wall, U=1 at edge).
        B[0] = 1.0; C_diag[0] = 0.0; D[0] = 0.0
        A[-1] = 0.0; B[-1] = 1.0; D[-1] = 1.0
        
        # Concatenate single 'diagonals' into 3 diagonals, does not require full matrix to solve as TDM, rest are 0.
        ab = np.zeros((3, N))
        ab[0, 1:] = C_diag[0:-1]
        ab[1, :] = B
        ab[2, 0:-1] = A[1:]
        
        return solve_banded((1, 1), ab, D)

    def solve_energy(self, U, f, f_i_m_one, g_i_m_one, C, Pr, H_wall_on_H_e, u_e, H_e, xi, delta_xi) -> np.ndarray:
        """Solves (C/Pr g')' + f g' + u_e^2/H_e [C(1 - 1/Pr) U U']'= 2xi(U' dg/dxi - g' df/dxi)"""
        N = config.n_y_cells
        
        # Calculate some additional parameters.
        E = C / Pr
        E_dash = np.gradient(E, self.delta_eta)
        U_dash = np.gradient(U, self.delta_eta)
        
        # Calculate viscous dissipation source term.
        V_core = C * (1.0 - 1.0 / Pr) * U * U_dash
        V_dash = np.gradient(V_core, self.delta_eta)
        V = (u_e**2 / H_e) * V_dash

        # Create arrays.
        A = np.zeros(N)
        B = np.zeros(N)
        C_diag = np.zeros(N)  # Call C_diag to avoid conflict with C for Chapmin-Rubesin constant.
        D = np.zeros(N)
        
        # Matrix coeffeicients to solve TDMA (a_j * U_i,j-1) + (b_j * U_i,j) + (c_j * U_i,j+1) = d_j.
        # P,Q,R,S have been substituted in directly to avoid creation of additional arrays.
        A[1:-1] = E[1:-1] / (self.delta_eta**2) - (E_dash[1:-1] + f[1:-1] + 2 * xi / delta_xi * (f[1:-1] - f_i_m_one[1:-1])) / (2 * self.delta_eta)
        B[1:-1] = -2 * E[1:-1] / (self.delta_eta**2) - 2 * xi / delta_xi * U[1:-1]
        C_diag[1:-1] = E[1:-1] / (self.delta_eta**2) + (E_dash[1:-1] + f[1:-1] + 2 * xi / delta_xi * (f[1:-1] - f_i_m_one[1:-1])) / (2 * self.delta_eta)
        D[1:-1] = -V[1:-1] - 2 * xi / delta_xi * U[1:-1] * g_i_m_one[1:-1] 
         
        # Boundary conditions (g = H_wall/H_e at wall, g = 1 at edge).
        B[0] = 1.0; C_diag[0] = 0.0; D[0] = H_wall_on_H_e
        A[-1] = 0.0; B[-1] = 1.0; D[-1] = 1.0

        if config.adiabatic:  # g[0] - g[1] = 0.
            C_diag[0] = -1.0; D[0] = 0.0
        
        # Concatenate single 'diagonals' into 3 diagonals, does not require full matrix to solve as TDM, rest are 0.
        ab = np.zeros((3, N))
        ab[0, 1:] = C_diag[0:-1]
        ab[1, :] = B
        ab[2, 0:-1] = A[1:]
        
        return solve_banded((1, 1), ab, D)

    def solve_species(self, U, f, f_i_m_one, s, s_i_m_one, omega_dot, C, Pr, rho_e_on_rho, rho_e, u_e, mu_e, s_e, s_e_eq, xi, delta_xi, r) -> np.ndarray:
        """Solves (C/Pr s_i')' + f s_i' + 2xi omega_dot/(rho rho_e mu_e u_e^2 r^(2k)) = 2xi(U' ds_i/dxi - s_i' df/dxi)"""
        N = config.n_y_cells
        
        # Calculate some additional parameters.
        E = C / Pr
        E_dash = np.gradient(E, self.delta_eta)
        
        # Calculate thermochemistry tource term.
        W_scale = (2 * xi * rho_e_on_rho) / (rho_e**2 * mu_e * u_e**2 * r**(2 * self.k))

        C_chem = np.where(omega_dot > 0, omega_dot, 0.0)
        s_safe = np.maximum(s, 1e-16)
        D_chem = np.where(omega_dot <= 0, -omega_dot / s_safe, 0.0)

        # Create arrays.
        A = np.zeros(N)
        B = np.zeros(N)
        C_diag = np.zeros(N)  # Call C_diag to avoid conflict with C for Chapmin-Rubesin constant.
        D = np.zeros(N)
        
        # Matrix coeffeicients to solve TDMA (a_j * U_i,j-1) + (b_j * U_i,j) + (c_j * U_i,j+1) = d_j.
        # P,Q,R,S have been substituted in directly to avoid creation of additional arrays.
        A[1:-1] = E[1:-1] / (self.delta_eta**2) - (E_dash[1:-1] + f[1:-1] + 2 * xi / delta_xi * (f[1:-1] - f_i_m_one[1:-1])) / (2 * self.delta_eta)
        B[1:-1] = -2 * E[1:-1] / (self.delta_eta**2) - (W_scale[1:-1] * D_chem[1:-1] + 2 * xi / delta_xi * U[1:-1])
        C_diag[1:-1] = E[1:-1] / (self.delta_eta**2) + (E_dash[1:-1] + f[1:-1] + 2 * xi / delta_xi * (f[1:-1] - f_i_m_one[1:-1])) / (2 * self.delta_eta)
        D[1:-1] = -W_scale[1:-1] * C_chem[1:-1] - 2 * xi / delta_xi * U[1:-1] * s_i_m_one[1:-1] 
         
        # Boundary conditions (s_i[0] - s_i[1] = 0 at wall, s_i = s_e at edge).
        B[0] = 1.0; C_diag[0] = -1.0; D[0] = 0.0
        A[-1] = 0.0; B[-1] = 1.0; D[-1] = s_e
        
        if config.catalytic:  # s_i = s_e at wall.
             C_diag[0] = 0.0; D[0] = s_e_eq

        # Concatenate single 'diagonals' into 3 diagonals, does not require full matrix to solve as TDM, rest are 0.
        ab = np.zeros((3, N))
        ab[0, 1:] = C_diag[0:-1]
        ab[1, :] = B
        ab[2, 0:-1] = A[1:]
        
        return solve_banded((1, 1), ab, D)

    def solve_system(self, x, u_e, T_e, P_e, rho_e, mu_e, h_e, s_e, H_wall, dp_e_dx, xi, delta_xi, r, prev_profiles=None) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray]:
        """Executes the Picard Iteration loop for a single x."""
        # 1. Evaluate additional parameters.
        H_e = h_e + 0.5 * u_e**2
        H_wall_on_H_e = H_wall / H_e
        beta = -2 * xi * dp_e_dx / (u_e**3 * rho_e**2 * mu_e)

        # 2. Initialise profiles.
        N = config.n_y_cells
        nsp = config.gmodel.n_species
        if prev_profiles is None:
            # Linear guess for first iteration.
            T_prof = T_e * np.ones(N) 
            U = self.eta / config.eta_end
            f = cumulative_trapezoid(U, self.eta, initial=0)
            g = H_wall_on_H_e + (1.0 - H_wall_on_H_e) * U
            if config.adiabatic:
                g = np.ones(N)

            s = np.empty([nsp, N])
            for i in range(nsp):
                s[i, :] = self.working_state.massf[i]
 
            U_i_m_one = U
            g_i_m_one = g
            f_i_m_one = f 
            s_i_m_one = s
            xi_copy = xi
            xi = 0.0

        else:
            U, g, f, s, T_prof = prev_profiles
            U_i_m_one, g_i_m_one, f_i_m_one, s_i_m_one, _ = prev_profiles

        # Memory allocations for fluid properties.
        k = np.zeros(N)
        mu = np.zeros(N)
        Cp = np.zeros(N)
        h_isp = np.zeros((N, nsp))
        C = np.zeros(N)
        Pr = np.zeros(N) 
        rho_e_on_rho = np.zeros(N)
        omega_dot = np.zeros([nsp, N])

        # Allocate pressure to working state.
        self.working_state.p = P_e

        # 3. Picard iteration loop.
        for iteration in range(config.max_iters):
            U_old = np.copy(U)
            g_old = np.copy(g)
            s_old = np.copy(s)
            
            # A. Calculate gas properties.
            for i in range(N):
                # Calculate local static enthalpy.
                h_local = H_e * g[i] - 0.5 * (u_e * U[i])**2
                
                # Update state using get_T.
                T_prof[i] = max(self.get_T(h_local, T_prof[i]), 100)  # Limit temperature to ensure GasState doesnt become invalid.
                self.working_state.T = T_prof[i] 
                self.working_state.massf = (s[:, i] / np.sum(s[:,i])).tolist()  # Ensure massf sum to 1.0.
                self.working_state.update_thermo_from_pT()
                self.working_state.update_trans_coeffs()
                
                # Extract parameters.
                k[i] = self.working_state.k
                mu[i] = self.working_state.mu
                C[i] = (self.working_state.rho * self.working_state.mu) / (rho_e * mu_e)
                Pr[i] = config.gmodel.Prandtl(self.working_state)
                rho_e_on_rho[i] = rho_e / self.working_state.rho

                if config.catalytic:
                    Cp[i] = self.working_state.Cp
                    for j in range(nsp):
                        h_isp[i, j] = self.working_state.enthalpy_isp(j)

                if config.reacting:
                    omega_dot_list = config.reactor.source_terms(self.working_state)
                    for j, val in enumerate(omega_dot_list):
                        omega_dot[j, i] = val

            # B. Solve linearised momentum.
            U_new = self.solve_momentum(U_old, U_i_m_one, f, f_i_m_one, C, beta, rho_e_on_rho, xi, delta_xi) 
            U = config.omega * U_new + (1 - config.omega) * U_old
            
            # C. Update stream function.
            f = cumulative_trapezoid(U, self.eta, initial=0)
            
            # D. Solve linearised energy.
            g_new = self.solve_energy(U, f, f_i_m_one, g_i_m_one, C, Pr, H_wall_on_H_e, u_e, H_e, xi, delta_xi)
            g = config.omega * g_new + (1 - config.omega) * g_old
            
            # E. Solve linearised species.
            if config.reacting:
                s_e_eq = self.get_EQ(P_e, T_prof[0], s_e)

                s_new = []
                for i in range(nsp):
                    s_new = self.solve_species(U, f, f_i_m_one, s[i], s_i_m_one[i], omega_dot[i], C, Pr, rho_e_on_rho, rho_e, u_e, mu_e, s_e[i], s_e_eq[i], xi, delta_xi, r)
                    s[i] = config.omega * s_new + (1 - config.omega) * s_old[i]

            # F. Check convergence.
            error_U = np.max(np.abs(U - U_old))
            error_g = np.max(np.abs(g - g_old))
            error_s = np.max([np.abs(s[i] - s_old[i]) for i in range(nsp)])
            max_error = max(error_U, error_g, error_s)
            
            if max_error < config.tol:
                if config.verbose:
                    print(f"x = {x:.5f}, WC = {(time() - self.start):.2f}, converged = y, iter = {iteration}, residual = {max_error:.2e}")
                break

        else:
            if config.verbose:
                print(f"x = {x:.5f}, WC = {(time() - self.start):.2f}, converged = n, iter = {iteration}, residual = {max_error:.2e}")

        # 4. Post-Process physical coordinates.
        if xi == 0.0:
            xi = xi_copy

        y = np.sqrt(2 * xi) / (u_e * rho_e * r**self.k) * cumulative_trapezoid(rho_e_on_rho, self.eta, initial=0)
        delta_star = np.sqrt(2 * xi) / (u_e * rho_e * r**self.k) * trapezoid(rho_e_on_rho - U, self.eta)
        
        return (U, g, f, s, T_prof), (k, mu, Cp, h_isp, Pr), y, delta_star, rho_e_on_rho

    def run(self) -> tuple[np.ndarray, ...]:
        """Main function to march through space and calculate the full boundary layer."""
        current_profiles = None

        # Initialise some arrays to store data.
        arrs = [np.zeros((config.n_x_cells, config.n_y_cells)) for _ in range(8)]
        
        y_arr, U_arr, rho_arr, T_arr, k_arr, mu_arr, Cp_arr, Pr_arr = arrs

        eta_arr = np.tile(self.eta, (config.n_x_cells, 1))
        delta_star_arr = np.zeros(config.n_x_cells)
        bl_height_arr = np.zeros(config.n_x_cells)
        massf_arr = np.zeros((config.n_x_cells, config.gmodel.n_species, config.n_y_cells))
        h_isp_arr = np.zeros((config.n_x_cells, config.n_y_cells, config.gmodel.n_species))

        if config.verbose:
            print("\nBeginning to march through space...\n")
            
        for i in range(config.n_x_cells):    
            current_profiles, transport_prop, y, delta_star, rho_e_on_rho = self.solve_system(self.x_arr[i], self.u_e_arr[i], self.T_e_arr[i], self.P_e_arr[i], 
                                                                            self.rho_e_arr[i], self.mu_e_arr[i], self.h_e_arr[i], self.massf_e_arr[i],
                                                                            self.H_wall_arr[i], self.dp_e_dx_arr[i], self.xi_arr[i], self.delta_xi_arr[i],
                                                                            self.r_arr[i], prev_profiles=current_profiles)              
                                                                         
            y_arr[i, :] = y; delta_star_arr[i] = delta_star; U_arr[i, :] = current_profiles[0] * self.u_e_arr[i]; massf_arr[i, :] = current_profiles[3]
            rho_arr[i, :] = self.rho_e_arr[i] / rho_e_on_rho; T_arr[i, :] = current_profiles[4]; k_arr[i, :] = transport_prop[0]; mu_arr[i, :] = transport_prop[1]
            Cp_arr[i, :] = transport_prop[2]; h_isp_arr[i, :] = transport_prop[3]; Pr_arr[i, :] = transport_prop[4]
         
            bl_idx = np.argmax(current_profiles[0] >= 0.99)
            bl_height_arr[i] = y[bl_idx] if current_profiles[0][bl_idx] >= 0.99 else np.nan
       
        massf_arr = np.swapaxes(massf_arr, 1, 2)
        tau_w = self.get_tau_w(mu_arr, U_arr,y_arr)
        q_c = self.get_q_c(k_arr, T_arr, y_arr)
        q_d = self.get_q_d(k_arr, Cp_arr, massf_arr, h_isp_arr, y_arr) if config.catalytic else np.zeros(config.n_x_cells)

        if config.verbose:
            print(f"\nSimulation finished.\n")        

        if config.export:
            self.to_vtk(y_arr, U_arr, massf_arr, rho_arr, T_arr, k_arr, mu_arr)
            np.savetxt("loads.dat", np.column_stack((self.x_arr, bl_height_arr, delta_star_arr, self.P_e_arr, tau_w, q_c, q_d)), delimiter=',', header="x (m), d (m), d* (m), p (Pa), tau (Pa), q_c (W/m^2), q_d (W/m^2)", comments="")

        return eta_arr, y_arr, bl_height_arr, delta_star_arr, U_arr, massf_arr, rho_arr, T_arr, k_arr, mu_arr, tau_w, q_c, q_d

    @staticmethod
    def get_tau_w(mu_arr, u_arr, y_arr) -> np.ndarray:
        """Calculates shear stress at the wall."""
        mu_cen = (mu_arr[:, 0] + mu_arr[:, 1]) / 2
        tau_w = mu_cen * u_arr[:, 1] / y_arr[:, 1]  # u and y = 0.0 @ wall.
        return tau_w

    @staticmethod
    def get_q_c(k_arr, T_arr, y_arr) -> np.ndarray:
        """Calculate conductive heat flux at the wall."""
        k_cen = (k_arr[:, 0] + k_arr[:, 1]) / 2
        q_c = k_cen * (T_arr[:, 1] - T_arr[:, 0]) / y_arr[:, 1]  # y = 0.0 @ wall.
        return q_c

    @staticmethod
    def get_q_d(k_arr, Cp_arr, massf_arr, enthalpy_isp_arr, y_arr) -> np.ndarray:
        """Calculate the species diffusive heat flux at the wall."""
        k_cen = (k_arr[:, 0] + k_arr[:, 1]) / 2
        Cp_cen = (Cp_arr[:, 0] + Cp_arr[:, 1]) / 2
        h_cen = (enthalpy_isp_arr[:, 0] + enthalpy_isp_arr[:, 1]) / 2  

        D_s = k_cen / Cp_cen  # rho not included as will cancel out in next step.

        massf_diff = massf_arr[:, 1] - massf_arr[:, 0]
        species_heat_fluxes = h_cen * (D_s[:, np.newaxis] * massf_diff) / y_arr[:, 1, np.newaxis]  # y = 0.0 @ wall.

        q_d = np.sum(species_heat_fluxes, axis=1)  

        return q_d

    def to_vtk(self, y_arr, U_arr, massf_arr, rho_arr, T_arr, k_arr, mu_arr) -> None:
            """Exports data to a vtk format."""
            nx = config.n_x_cells

            ny_original = config.n_y_cells
            ny_new = ny_original + 1 
            y_max_global = np.max(y_arr)

            X_2d = np.zeros((nx, ny_new))
            Y_2d = np.zeros((nx, ny_new))
            Z_2d = np.zeros_like(X_2d)
            U_2d = np.zeros((nx, ny_new))
            T_2d = np.zeros((nx, ny_new))
            rho_2d = np.zeros((nx, ny_new))
            k_2d = np.zeros((nx, ny_new))
            mu_2d = np.zeros((nx, ny_new))
            Massf_2d = np.zeros((nx, ny_new, config.gmodel.n_species))

            Massf_2d[:, :-1, :] = massf_arr
            Massf_2d[:, -1, :] = massf_arr[:, -1, :]
            
            X_2d[:, :-1] = np.tile(self.x_arr, (ny_original, 1)).T
            X_2d[:, -1] = np.tile(self.x_arr, (ny_original, 1)).T[:, -1]

            Y_2d[:, :-1] = y_arr
            Y_2d[:, -1] = y_max_global
 
            U_2d[:, :-1] = U_arr
            U_2d[:, -1] = U_arr[:, -1]

            T_2d[:, :-1] = T_arr
            T_2d[:, -1] = T_arr[:, -1]

            rho_2d[:, :-1] = rho_arr
            rho_2d[:, -1] = rho_arr[:, -1]

            k_2d[:, :-1] = k_arr
            k_2d[:, -1] = k_arr[:, -1]           

            mu_2d[:, :-1] = mu_arr
            mu_2d[:, -1] = mu_arr[:, -1]

            grid = pv.StructuredGrid(X_2d, Y_2d, Z_2d)

            grid.point_data["vel.x"] = U_2d.flatten(order='F')
            grid.point_data["T"] = T_2d.flatten(order='F')
            grid.point_data["rho"] = rho_2d.flatten(order='F')
            grid.point_data["k"] = k_2d.flatten(order='F')
            grid.point_data["mu"] = mu_2d.flatten(order='F')

            for i in range(config.gmodel.n_species):
                grid.point_data["massf-" + config.gmodel.species_names[i]] = Massf_2d[:, :, i].flatten(order='F')
    
            filename = "output.vts"
            grid.save(filename)
            
            if config.verbose:
                print(f"Export complete.\n")

class ViscousInteraction():
    """Solves for the viscous interaction over a flat plate or wedge."""
    def __init__(self, u, p, T, massf, T_wall=300.0, x_start=0.00001, x_end=0.1, theta=0.0, tol=1.0, d_star_idx=100) -> None:
        # Some initital setup.
        global config
        global active_vi

        active_vi = self

        # Inflow conditions.
        self.u = u
        self.p = p
        self.T = T
        self.massf = massf
        self.T_wall = T_wall

        # Some additional user parameters.
        self.theta_wedge = np.deg2rad(theta)
        self.tol = tol
        self.d_star_idx = d_star_idx
        self.x_start = x_start
        self.x_end = x_end
    
    @staticmethod
    def pressure_model(x, a, b, c) -> float:
        """Function that the pressure distribution follows."""
        return a / np.sqrt(np.abs(x) + c) + b
    
    def get_egde(self, x, d_star, state1, u1, state2, flow) -> tuple[np.ndarray, ...]:
        """Calculate boundary-layer edge properties using tangent wedge method."""
        x = np.insert(x, 0, 0); d_star = np.insert(d_star, 0, 0)

        n = min(self.d_star_idx, config.n_x_cells)  # Prevent micro-oscillations (due to numerics) from showing up in the pc curve which causes invalid theta vals and crashes theta-beta-M.
        indices = [i * len(x) // n for i in range(n)]  # By only taking n amount of points to define the pc curve, you ensure it is smooth and monotonic while preserving the overall shape.
        pc = PchipInterpolator(x[indices], d_star[indices])  # Then using the actual amount of len(x) points still gives desired accuracy.
        theta = pc.derivative() 

        p = np.full_like(x, state1.p)
        T = np.full_like(x, state1.T)
        u = np.full_like(x, u1)

        for idx, val in enumerate(x):
            beta = flow.beta_oblique(state1, u1, max(np.arctan(theta(val)), 0.0))
            _, u2 = flow.theta_oblique(state1, u1, beta, state2)
            p[idx] = state2.p
            T[idx] = state2.T
            u[idx] = u2
        
        p_idx = np.argmax(p)
        x_train = x[p_idx:]
        p_train = p[p_idx:]
        p_opt, _ = curve_fit(self.pressure_model, x_train, p_train, bounds=([-np.inf, -np.inf, 1e-10], [np.inf, np.inf, np.inf]), p0=[1.0, 0.0, 1e-6], maxfev=10000)
        p = self.pressure_model(x, *p_opt)

        return x[1:], p[1:], uniform_filter1d(T[1:], size=10, mode='nearest'), uniform_filter1d(u[1:], size=10, mode='nearest')  # type: ignore

    def get_post_shock(self, state1, state2, flow) -> tuple[float, ...]:
        """Calcultae post shock properties using oblique shock relations."""
        state1.p = self.p
        state1.T = self.T
        state1.massf = self.massf
        
        state1.update_thermo_from_pT()

        beta = flow.beta_oblique(state1, self.u, self.theta_wedge)

        _, u2 = flow.theta_oblique(state1, self.u, beta, state2)

        return u2, state2.p, state2.T

    def run(self) -> tuple[np.ndarray, ...]:
        """Calculates the viscous interaction over a flat plate using crbl and the tangent-wedge method."""
        # Initialise inflow GasState.
        state1 = GasState(config.gmodel)
        state2 = GasState(config.gmodel)
        flow = GasFlow(config.gmodel)

        if self.theta_wedge > 0.0:
            self.u, self.p, self.T = self.get_post_shock(state1, state2, flow)

        state1.p = self.p
        state1.T = self.T
        state1.massf = self.massf
        state1.update_thermo_from_pT()

        # Guess some initial profiles.
        x = np.linspace(self.x_start, self.x_end, config.n_x_cells)
        P_e = self.p * (1 + 0.1 / np.sqrt(x))
        u_e = np.full(config.n_x_cells, self.u)
        T_e = np.full(config.n_x_cells, self.T)
        mass_f_e = np.full((config.n_x_cells, config.gmodel.n_species), self.massf)

        # Setup loop.
        P_e_old = P_e.copy()
        err = 10 * self.tol
        iters = 0

        # Compute viscous-inviscid interaction.
        while err > self.tol and iters < 30:   
            bl = BoundaryLayer(x_arr=x, u_e_arr=u_e, T_e_arr=T_e, P_e_arr=P_e, massf_e_arr=mass_f_e, T_wall=self.T_wall)

            _, _, _, d_star, _, _, _, _, _, _, tau_w, q_c, q_d = bl.run()
            
            try:
                x, P_e, T_e, u_e = self.get_egde(x, d_star, state1, self.u, state2, flow)  # Use Tangent-Wedge method to calculate pressure distribution.
            
            except Exception as e:
                print(f"Tangent-wedge method failed : {e}")
                break
            
            err = np.max(np.abs(P_e - P_e_old))
            P_e_old = P_e.copy()

            iters += 1

            if config.verbose:
                print(f"Vii Iteration = {iters}, pressure residual = {err:.2e}")
        
        if config.verbose:
            print()

        return x, d_star, P_e, tau_w, q_c, q_d 

def read_inputs() -> tuple[str, str]:
    """Read CLI inputs and print help message."""
    crbl_input = argv[1] if len(argv) > 1 else ""
                
    if crbl_input == "--help" or crbl_input == "":
        print("\n|   ____ ____  ____  _     ")
        print(r"|  / ___|  _ \| __ )| |")
        print(r"| | |   | |_) |  _ \| |   v" + __version__)
        print(r"| | |___|  _ <| |_) | |___")
        print(r"|  \____|_| \_\____/|_____|")
        print("|")
        print("| Chemically Reacting Boundary-Layer")
        print("\nUsage: crbl [options]\n")
        print("Options:")
        print("    --help   Displays help message and exits.")
        print("    --bl=    Initialise a boundary-layer simulation using a .py configuration file.")
        print("    --vi=    Initialise a viscous interaction simulation over a flate plate or wedge using a .py configuration file.\n")
        quit()

    inputs = crbl_input.split("=")
    job_type = inputs[0]
    path = inputs[1]

    return job_type, path

def load_config(path: str) -> BoundaryLayer | ViscousInteraction:
    """Loads the configuration from a seperate .py file."""
    global active_bl
    global active_vi
    
    with open(path, "r") as f:
        exec(f.read(), globals())
    
    if active_bl is not None:
        return active_bl
    
    if active_vi is not None:
        return active_vi

    raise RuntimeError("Configuration file must initialise a BoundaryLayer() or Vii() object.")

def main() -> None:
    job_type, path = read_inputs()

    try:
        if job_type == '--bl':
            bl = load_config(path)
            bl.run()

        if job_type == '--vi':
            vi = load_config(path)
            vi.run()

    except Exception as e:
        print(e) 

config = GlobalConfig()
active_bl = None
active_vi = None

if __name__ == '__main__':    
    main()