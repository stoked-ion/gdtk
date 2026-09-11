# type:ignore
from libc.math cimport fabs, sqrt, fmax, fmin, tanh, log
from libc.stdlib cimport malloc, free

cdef void beta_ivp(double[:] u, double[:] x_cen, double[:] rho, double[:] p, 
                   double DOMAIN, double RADIUS, double R_s, int shock_idx, 
                   double rho_frozen, double[:] beta) noexcept nogil:
    """
    Solves the inviscid ivp in non-conservative form using implicit backward Euler integration.
    
    rho*u*dbeta/dx + rho*beta^2 = -d2p/dy2

    """

    cdef double dx_cen_diff, d2p_dy2_next, S_next, B, C, discriminant
    cdef double rho_inf = rho[0]
    cdef double u_inf = u[0]
    cdef double p_inf = p[0]
    cdef int n = u.shape[0]
    cdef double R_eff
    cdef int i

    cdef double beta_s = (u_inf / R_s) * (1.0 - (rho_inf / rho_frozen))
    cdef double L = DOMAIN - x_cen[shock_idx]
    beta[shock_idx] = beta_s

    for i in range(shock_idx, n - 1):
        dx_cen_diff = x_cen[i+1] - x_cen[i]

        R_eff = RADIUS * 0.866 + (R_s - RADIUS * 0.866) * (DOMAIN - x_cen[i + 1]) / L
        d2p_dy2_next = -2.0 * (p[i + 1] - p_inf) / R_eff**2
        S_next = -(1.0 / rho[i + 1]) * d2p_dy2_next
        
        B = u[i + 1] / dx_cen_diff
        C = -(B * beta[i] + S_next)
        
        discriminant = fmax(B**2 - 4.0 * C, 0.0)
        
        beta[i + 1] = (-B + sqrt(discriminant)) / 2.0

    for i in range(0, shock_idx):
        beta[i] = 0.0

cdef void beta_bvp(double[:] u, double[:] x_cen, double[:] rho, double[:] p, double[:] mu, 
                   double DOMAIN, double RADIUS, double R_s, int shock_idx, 
                   double rho_frozen, double[:] beta) noexcept nogil:
    """
    Solves the viscous bvp in non-conservative form for beta using Finite differences + Picard Iteration + TDMA.
    
    rho*u*dbeta/dx + rho*beta^2 = -d2p/dy2 + d/dx(mu*dbeta/dx)

    """

    cdef double dx_f, dx_b, dx_c, S_i, error, max_error, diff, mu_f, mu_b
    cdef int n = x_cen.shape[0] - shock_idx
    cdef double R_eff
    cdef int i, iter
    
    # Extract freestream conditions from index 0.
    cdef double rho_inf = rho[0]
    cdef double p_inf = p[0]
    cdef double u_inf = u[0]
    
    # Calculate initial boundary condition at the shock.
    cdef double beta_s = (u_inf / R_s) * (1.0 - (rho_inf / rho_frozen))
    
    # Allocate temporary arrays for TDMA.
    cdef double* A = <double*>malloc(n * sizeof(double))
    cdef double* B = <double*>malloc(n * sizeof(double))
    cdef double* C = <double*>malloc(n * sizeof(double))
    cdef double* D = <double*>malloc(n * sizeof(double))
    cdef double* c_star = <double*>malloc(n * sizeof(double))
    cdef double* d_star = <double*>malloc(n * sizeof(double))
    cdef double* beta_new = <double*>malloc(n * sizeof(double))

    cdef double x_shock = x_cen[shock_idx]
    cdef double L = DOMAIN - x_shock
    
    # Initial guess.
    for i in range(n):
        beta_new[i] = beta_s * (1.0 - (x_cen[shock_idx + i] - x_shock) / L)

    # Picard Iteration Loop. 
    error = 1.0
    iter = 0
    while error > 1e-6 and iter < 100:
        
        # 1. Build Tridiagonal System.
        for i in range(1, n - 1):
            dx_f = x_cen[shock_idx + i + 1] - x_cen[shock_idx + i]
            dx_b = x_cen[shock_idx + i] - x_cen[shock_idx + i - 1]
            dx_c = dx_f + dx_b 
            
            R_eff = RADIUS * 0.866 + (R_s - RADIUS * 0.866) * (DOMAIN - x_cen[shock_idx + i]) / L
            S_i = -2.0 * (p[shock_idx + i] - p_inf) / R_eff**2

            mu_f = 0.5 * (mu[shock_idx + i] + mu[shock_idx + i + 1])
            mu_b = 0.5 * (mu[shock_idx + i - 1] + mu[shock_idx + i])

            A[i] = (2.0 * mu_b) / (dx_c * dx_b) + (rho[shock_idx + i] * u[shock_idx + i]) / dx_c
            B[i] = -(2.0 / dx_c) * (mu_f / dx_f + mu_b / dx_b) - (rho[shock_idx + i] * beta_new[i])
            C[i] = (2.0 * mu_f) / (dx_c * dx_f) - (rho[shock_idx + i] * u[shock_idx + i]) / dx_c
            D[i] = S_i

        D[1] -= A[1] * beta_s
        
        # Apply no-slip wall BC, beta_ghost = -beta_interior.
        B[n-2] -= C[n-2]
        C[n-2] = 0.0

        # 2. TDMA Forward Sweep
        c_star[1] = C[1] / B[1]
        d_star[1] = D[1] / B[1]
        for i in range(2, n - 1):
            c_star[i] = C[i] / (B[i] - A[i] * c_star[i - 1])
            d_star[i] = (D[i] - A[i] * d_star[i - 1]) / (B[i] - A[i] * c_star[i - 1])

        # 3. TDMA Back Substitution.
        max_error = 0.0
        for i in range(n - 2, 0, -1):
            diff = beta_new[i]
            beta_new[i] = d_star[i] - c_star[i] * beta_new[i + 1]
            diff = fabs(beta_new[i] - diff)
            if diff > max_error:
                max_error = diff
                
        beta_new[n - 1] = -beta_new[n - 2]
        beta_new[0] = beta_s 
        error = max_error
        iter += 1

    # Write back to memoryview.
    for i in range(n):
        beta[shock_idx + i] = beta_new[i]
        
    for i in range(0, shock_idx):
        beta[i] = 0.0

    free(A); free(B); free(C); free(D); free(c_star); free(d_star); free(beta_new)

cdef void smooth_beta(double[:] x_cen, double[:] p, double[:] beta, int shock_idx) noexcept nogil:
    """Apply smoothing to beta to account for a diffusive shock of finite thickness."""

    cdef double grad_tol_frac = 0.01
    cdef double min_delta_A = 1e-10

    cdef double a, max_dpdx, dpdx, grad_tol, delta_A, k, u
    cdef int i, start_idx, end_idx
    cdef int n = x_cen.shape[0]

    if shock_idx <= 0 or shock_idx >= n - 1:
        return

    a = (x_cen[shock_idx - 1] + x_cen[shock_idx]) / 2.0

    max_dpdx = 0.0
    for i in range(1, n - 1):
        dpdx = (p[i] - p[i - 1]) / (x_cen[i] - x_cen[i - 1])

        if dpdx > max_dpdx:
            max_dpdx = dpdx

    if max_dpdx <= 0.0:
        return

    grad_tol = grad_tol_frac * max_dpdx

    start_idx = shock_idx - 1
    for i in range(shock_idx - 1, 0, -1):
        dpdx = (p[i] - p[i - 1]) / (x_cen[i] - x_cen[i - 1])

        if dpdx < grad_tol:  
            start_idx = i
            break

    if start_idx == shock_idx - 1:
        delta_A = min_delta_A

    else:
        delta_A = a - x_cen[start_idx]

    if delta_A < min_delta_A:
        delta_A = min_delta_A

    end_idx = shock_idx
    while end_idx < n and (x_cen[end_idx] - a) <= 3.0 * delta_A:
        end_idx += 1

    k = 15 / (8 * delta_A)

    for i in range(start_idx, shock_idx):
        u = fmax(fmin(0.5 * (x_cen[i] - (x_cen[shock_idx] - delta_A)) / delta_A, 1.0), 0.0)

        beta[i] = beta[shock_idx] * (u * u * u * (u * (6.0 * u - 15.0) + 10.0))
        
    for i in range(shock_idx, end_idx):
        beta[i] = beta[i] * 0.5 * (1.0 + tanh(k * (x_cen[i] - x_cen[shock_idx])))
