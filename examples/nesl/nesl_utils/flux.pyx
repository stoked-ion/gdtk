# type: ignore
from libc.math cimport fmin, fmax
cimport cython

cdef inline void hllc(double uL, double rhoL, double pL, double eL, double e_vrL, double aL,
                      double[:] mass_fracL, double uR, double rhoR, double pR, double eR, double e_vrR, 
                      double aR, double[:] mass_fracR, int nsp, int nmodes, double[:] flux) noexcept nogil:
    """Calculates the convective flux across an interface accounting for different thermodynamic 
       states using the Harten-Lax-van Leer-Contact approximate Riemann Solver."""

    cdef double E_L
    cdef double E_R
    cdef double S_L
    cdef double S_R
    cdef double S_star
    cdef double rho_star
    cdef double E_star
    cdef int i

    # Step 1: Energy and Wave speed estimates.
    E_L = eL + 0.5 * uL**2
    E_R = eR + 0.5 * uR**2

    # Step 2: Wave speed estimates.
    S_L = fmin(uL - aL, uR - aR)
    S_R = fmax(uL + aL, uR + aR)
    
    # Step 3: Star region
    S_star = (pR - pL + rhoL * uL * (S_L - uL) - rhoR * uR * (S_R - uR)) / \
             (rhoL * (S_L - uL) - rhoR * (S_R - uR))
    
    # Step 4: Flux calculation based on wave regions.
    if S_L >= 0.0:
        # Left.
        for i in range(nsp):
            flux[i] = mass_fracL[i] * uL * rhoL

        flux[nsp] = rhoL * uL**2 + pL
        flux[nsp + 1] = (rhoL * E_L + pL) * uL
        flux[nsp + 2] = (rhoL * e_vrL) * uL

        if nmodes == 0:
            flux[nsp + 2] = 0.0

    elif S_R <= 0.0:
        # Right.
        for i in range(nsp):
            flux[i] = mass_fracR[i] * uR * rhoR

        flux[nsp] = rhoR * uR**2 + pR
        flux[nsp + 1] = (rhoR * E_R + pR) * uR
        flux[nsp + 2] = (rhoR * e_vrR) * uR

        if nmodes == 0:
            flux[nsp + 2] = 0.0

    elif S_star >= 0.0:
        # Left Star.
        rho_star = rhoL * (S_L - uL) / (S_L - S_star)
        E_star = E_L + (S_star - uL) * (S_star + pL / (rhoL * (S_L - uL)))

        for i in range(nsp):
            flux[i] = mass_fracL[i] * (rhoL * uL + S_L * (rho_star - rhoL))
        
        flux[nsp] = (rhoL * uL**2 + pL) + S_L * (rho_star * S_star - rhoL * uL)
        flux[nsp + 1] = (rhoL * E_L + pL) * uL + S_L * (rho_star * E_star - rhoL * E_L)
        flux[nsp + 2] = (rhoL * e_vrL) * uL + S_L * (rho_star * e_vrL - rhoL * e_vrL)

        if nmodes == 0:
            flux[nsp + 2] = 0.0

    else:
        # Right Star.
        rho_star = rhoR * (S_R - uR) / (S_R - S_star)
        E_star = E_R + (S_star - uR) * (S_star + pR / (rhoR * (S_R - uR)))
        
        for i in range(nsp):
            flux[i] = mass_fracR[i] * (rhoR * uR + S_R * (rho_star - rhoR))
            
        flux[nsp] = (rhoR * uR**2 + pR) + S_R * (rho_star * S_star - rhoR * uR)
        flux[nsp + 1] = (rhoR * E_R + pR) * uR + S_R * (rho_star * E_star - rhoR * E_R)
        flux[nsp + 2] = (rhoR * e_vrR) * uR + S_R * (rho_star * e_vrR - rhoR * e_vrR)

        if nmodes == 0:
            flux[nsp + 2] = 0.0
