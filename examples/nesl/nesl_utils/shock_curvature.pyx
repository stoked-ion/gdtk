# type: ignore
from libc.math cimport exp, fabs, pow, fmax

cdef void shock_curvature(double[:] u, double[:] x_cen, double[:] rho, double [:] dx, double DOMAIN, 
                          double RADIUS, str CURVATURE, double* R_s, double* standoff, 
                          int* shock_idx, int alpha) noexcept nogil:
    """Calculate the radius of curvature of the bow shock."""
    
    cdef double current_grad, rho_sum, rho_2_avg
    cdef double rho_inf = rho[0]
    cdef double max_grad = -1.0
    cdef int n = u.shape[0]
    cdef int idx = 0
    cdef int i
    
    for i in range(n - 1):
        current_grad = fabs((u[i + 1] - u[i]) / (x_cen[i + 1] - x_cen[i]))
        if current_grad > max_grad:
            max_grad = current_grad
            idx = i

    shock_idx[0] = idx
    standoff[0] = DOMAIN - x_cen[idx]

    if CURVATURE == "empirical":
        rho_sum = 0.0
        
        for i in range(idx, n - 1):
            rho_sum += rho[i] * dx[i]
        
        rho_2_avg = rho_sum / (standoff[0] + 0.5 * dx[idx])

        if alpha == 1:
            R_s[0] = RADIUS * (1 + 11.6935 / fmax(rho_2_avg / rho_inf - 1.3065, 0.0)**1.4735)  # Clamp solution to prevent NaN.

        else:
            R_s[0] = RADIUS * exp(1.1617 / pow(fmax(rho_2_avg / rho_inf - 1.005, 0.0), 1.096))  # Clamp solution to prevent NaN.

    else:
        R_s[0] = RADIUS + standoff[0]