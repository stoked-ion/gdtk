# type: ignore
from libc.math cimport fabs, fmin, fmax, INFINITY

cdef void compute_dt(double[:] u, double[:] a_st, double[:] rho_st, double[:] mu_st, double[:] k_st, double[:] Cp_st, 
                     double[:] dt, double[:] dx, double CFL_c, double CFL_d, double LEWIS_NUM) noexcept nogil:
    """Computes the stable diffusive and convective timestep."""

    cdef double u_val, alpha_val, nu_val, dx_val
    cdef double min_dx = INFINITY
    cdef double max_alpha = 0.0
    cdef double max_abs_u = 0.0
    cdef double max_nu = 0.0
    cdef double max_a = 0.0
    cdef int n = u.shape[0]
    cdef int i

    for i in range(1, n - 1):
        # Find max(abs(u)).
        u_val = fabs(u[i])
        if u_val > max_abs_u:
            max_abs_u = u_val

        # Find max(a_st).    
        if a_st[i] > max_a:
            max_a = a_st[i]
            
        # Find max(kinematic viscosity).
        nu_val = mu_st[i] / rho_st[i]
        if nu_val > max_nu:
            max_nu = nu_val

        # Find max(thermal diffusivity).
        alpha_val = k_st[i] / (Cp_st[i] * rho_st[i])
        if alpha_val > max_alpha:
            max_alpha = alpha_val
            
        # Find min(dx).
        dx_val = dx[i]
        if dx_val < min_dx:
            min_dx = dx_val

    cdef double dt_c = CFL_c * min_dx / (max_abs_u + max_a)    
    cdef double dt_d = CFL_d * (min_dx * min_dx) / fmax(max_nu, fmax(max_alpha, max_alpha / LEWIS_NUM))

    cdef double dt_min = fmin(dt_c, dt_d)

    for i in range(1, n):
        dt[i] = dt_min

cdef void compute_dt_lts(double[:] u, double[:] a_st, double[:] rho_st, double[:] mu_st, double[:] k_st, double[:] Cp_st, 
                         double[:] dt, double[:] dx, double CFL_c, double CFL_d, double LEWIS_NUM) noexcept nogil:
    """Computes the stable diffusive and convective timestep using local time stepping."""

    cdef double dt_d, dt_c, alpha_val
    cdef int n = u.shape[0]
    cdef int i

    for i in range(1, n - 1):
        alpha_val = k_st[i] / (Cp_st[i] * rho_st[i])

        dt_c = CFL_c * dx[i] / (fabs(u[i]) + a_st[i])
        dt_d = CFL_d * (dx[i] * dx[i]) / fmax(mu_st[i] / rho_st[i], fmax(alpha_val, alpha_val / LEWIS_NUM))

        dt[i] = fmin(dt_c, dt_d)

    dt[n - 1] = dt[n - 2]