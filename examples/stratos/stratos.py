"""
A script to simulate the post shock relaxation 
of 1T and 2T flows. Built on a space marching 
FVM scheme solving the steady 1D Euler equations.

Author: Taine J. Rossini (t.rossini@uq.edu.au)
Build Date: May 18, 2026

"""

__version__ = "1.0.0"

from gdtk.gas import GasModel, GasState, GasFlow, ThermochemicalReactor
from scipy.optimize import newton
import numpy as np

class Stratos():
    """Shocked Thermochemical Relaxation."""
    def __init__(self, gasFile, chemFile, kinFile, x_end, cells, u1, massf1, T1, p1=None, rho1=None) -> None:
        # User defined parameters.
        self.x_end = x_end
        self.u1 = u1
        self.cells = cells

        # Intialise GDTK objects.
        self.gmodel = GasModel(gasFile)
        self.reactor = ThermochemicalReactor(self.gmodel, chemFile, kinFile)
        self.flow = GasFlow(self.gmodel)
        self.state = GasState(self.gmodel)
        self.state1 = GasState(self.gmodel)
        self.state2 = GasState(self.gmodel)
        self.nmodes = self.gmodel.n_modes

        # Update state.
        self.state1.T = T1
        self.state1.massf = massf1
        self.state1.T_modes = [T1]

        if p1 is not None:
            self.state1.p = p1
            self.state1.update_thermo_from_pT()

        else:
            self.state1.rho = rho1
            self.state1.update_thermo_from_rhoT()
    
    @staticmethod
    def residual(u, state, C1, C2, C3) -> float:
        """Used in root finding to calculate u."""
        rho = C1 / u
        p = C2 - rho * u**2

        state.rho = rho
        state.p = p

        state.update_thermo_from_rhop()  # Requires arbritray guess for T_modes (from last step), and u_modes being set.

        return state.enthalpy + 0.5 * u**2 - C3

    @staticmethod
    def process_data(beta, u2_t, columns) -> None:
        columns['flow_angle'] = beta - np.arctan(columns['u'] / u2_t)

        columns['x_t'] = columns['t'] * u2_t  # Distance tangential (parallel) to straight shock.
        columns['x_n'] = columns['x']   # Distance normal to straight shock.
        columns['x_arc'] = np.insert(np.cumsum(np.sqrt(np.diff(columns['x_n'])**2 + np.diff(columns['x_t'])**2)), 0, 0.0)  # Distance along the streamline.

        columns['x_coord'] = columns['x_n'] * np.sin(beta) + columns['x_t'] * np.cos(beta)  # Transform from shock-aligned coordinates (n, t) to Cartesian (X, Y).
        columns['y_coord'] = -columns['x_n'] * np.cos(beta) + columns['x_t'] * np.sin(beta)

        columns['u_t'] = np.full_like(columns['u'], u2_t)  # Velocity tangential (parallel) to straight shock.
        columns['u_n'] = columns['u']  # Velocity normal to straight shock.
        columns['u'] = np.sqrt(u2_t**2 + columns['u_n']**2)  # Velocity vector at each point.

        del columns['x']

    @staticmethod
    def interpolate_to_length(no_cells, target_length, columns, reference) -> None:
        """Interpolates data onto an evenly spaced target length."""
        new_x = np.linspace(0, target_length, no_cells)
        
        old_x = columns[reference]
        
        for key in columns.keys():
            columns[key] = np.interp(new_x, old_x, columns[key])
    
    def march(self, state2, u2) -> dict[str, np.ndarray]:
        """Performs the space marching based on provided initital conditions."""
        # Define variables to avoid overhead of calling in hot loop.
        reactor = self.reactor
        state = self.state

        # Initial conditions.
        F_massf = [x * state2.rho * u2 for x in state2.massf]
        F_e_ve = state2.rho * state2.u_modes[0] * u2 if self.nmodes else 0.0
        massf = state2.massf
        state.T = state2.T 
        state.T_modes = [self.state1.T]
        u_guess = u2

        # Constant conserved variables.
        C1 = state2.rho * u2
        C2 = state2.rho * u2**2 + state2.p
        C3 = state2.enthalpy + 0.5 * u2**2

        # Make grid -> even though this is FVM, since we are working with fluxes directly, data is stored at cell interface not cell centre.
        x = np.linspace(0, self.x_end, self.cells)
        dx = self.x_end / self.cells

        # Make list of tuple to store data.
        base = ('x', 'u', 'T_tr', 'T_ve', 'p', 'rho', 'e_tr', 'e_ve', 'gamma', 'R') if self.nmodes else ('x', 'u', 'T_tr', 'p', 'rho', 'e_tr', 'gamma', 'R')
        props = [(*base, *(f'Y_{s}' for s in self.gmodel.species_names), *(f'X_{s}' for s in self.gmodel.species_names))]

        # Begin marching
        for i in range(len(x)): 
            massf = [x / C1 for x in F_massf]
            massf = [x / sum(massf) for x in massf]
            state.massf = massf  # Set here once instead of inside residual loop, remains constant and is only 'updated' through source terms / flux not thermo update.

            e_ve = F_e_ve / C1
            state.u_modes = [e_ve]  # Set here once instead of inside residual loop, remains constant and is only 'updated' through source terms / flux not thermo update.

            u = newton(self.residual, x0=u_guess, args=(state, C1, C2, C3))

            # We do not need to do an additional update_thermo_from_rhop() here as the converged value is still stored in state.

            u_guess = u

            source_terms = reactor.source_terms(state)

            F_massf = [F_massf[j] + source_terms[j] * dx for j in range(len(F_massf))]
            F_e_ve += source_terms[-1] * dx

            props.append((x[i], u, state.T, *state.T_modes, state.p, state.rho, state.u, *state.u_modes, state.gamma, state.R, *state.massf, *state.molef))  # type: ignore
        
        # Process the data
        headers = props[0]
        data = np.array(props[1:])
        columns = dict(zip(headers, data.T))

        columns['t'] = np.insert(np.cumsum(dx / columns['u'][:-1]), 0, 0.0)

        return columns

    def normal(self) -> dict[str, np.ndarray]:
        """Normal shock jump conditions."""
        u2, _ = self.flow.normal_shock(self.state1, self.u1, self.state2)

        return self.march(self.state2, u2)

    def oblique(self, theta, verbose=True, end_type='x_arc', max_iter=30, tol=1e-5) -> tuple[dict[str, np.ndarray], float]:
        """Oblique shock jump conditions with non-eq corrected shock angle using locally straight shock assumption."""
        beta = self.flow.beta_oblique(self.state1, self.u1, theta)
        
        x_orig = self.x_end

        beta_prev = None
        db_prev = None

        points = int(-np.log10(tol))

        # Begin iteration using Secant Method -> drive final flow angle to match theta.
        for _ in range(max_iter):

            _, u2 = self.flow.theta_oblique(self.state1, self.u1, beta, self.state2)

            u2_t = np.cos(beta) * self.u1
            u2_n = np.sqrt(u2**2 - u2_t**2)

            sin_alpha = u2_n / np.sqrt(u2_n**2 + u2_t**2)  # Frozen.
            self.x_end = x_orig * sin_alpha * 1.10   # This is just a guess.
            
            columns = self.march(self.state2, u2_n)

            last_flow_angle = beta - np.arctan(columns['u'][-1] / u2_t)

            # f(beta).
            db = last_flow_angle - theta

            if verbose:
                print(f"Residual: {db:.{points}f} | Last Flow Angle: {np.degrees(last_flow_angle):.{points}f} | Beta: {np.degrees(beta):.{points}f}")

            if abs(db) < tol:
                break

            # f'(beta).
            if beta_prev is None or (beta - beta_prev) == 0:
                derivative = 1.0 

            else:
                derivative = (db - db_prev) / (beta - beta_prev)

            beta_prev = beta
            db_prev = db

            # Secant update.
            beta = beta - (db / derivative)

        # Process the data
        self.process_data(beta, u2_t, columns)
        columns['x_p'] = columns['x_coord'] * np.cos(theta) + columns['y_coord'] * np.sin(theta)  # Distance parallel (tangential) to the wedge.
        self.interpolate_to_length(self.cells, x_orig, columns, end_type)
        
        self.x_end = x_orig
        
        return columns, beta

    def shock_angle(self, beta) -> dict[str, np.ndarray]:
        """Track the streamline behind a known shock angle."""
        x_orig = self.x_end

        u2_t = np.cos(beta) * self.u1
        u1_n = np.sin(beta) * self.u1

        u2_n, _ = self.flow.normal_shock(self.state1, u1_n, self.state2)

        sin_alpha = u2_n / np.sqrt(u2_n**2 + u2_t**2)  # Frozen. 
        self.x_end = x_orig * sin_alpha * 1.10   # This is just a guess.

        columns = self.march(self.state2, u2_n)

        # Process the data.
        self.process_data(beta, u2_t, columns)
        self.interpolate_to_length(self.cells, x_orig, columns, 'x_arc')

        self.x_end = x_orig

        return columns

if __name__ == '__main__':    
    print('Stratos main called.')
    print('Interact with the solver by importing the class into a .py file.')

