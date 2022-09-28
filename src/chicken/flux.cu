// flux.cu
// Include file for chicken.
// PJ 2022-09-11

#ifndef FLUX_INCLUDED
#define FLUX_INCLUDED

#include <string>
#include "number.cu"
#include "vector3.cu"
#include "gas.cu"
#include "flow.cu"
#include "face.cu"

using namespace std;

__host__ __device__
void ausmdv(FVFace& f, const FlowState& fsL, const FlowState& fsR)
// Compute the face's flux vector from left and right flow states.
// Wada and Liou's flux calculator, implemented from details in their AIAA paper,
// with hints from Ian Johnston.
// Y. Wada and M. -S. Liou (1994)
// A flux splitting scheme with high-resolution and robustness for discontinuities.
// AIAA-94-0083.
{
    Vector3 velL = Vector3(fsL.vel);
    Vector3 velR = Vector3(fsR.vel);
    velL.transform_to_local_frame(f.n, f.t1, f.t2);
    velR.transform_to_local_frame(f.n, f.t1, f.t2);
    //
    number rhoL = fsL.gas.rho;
    number pL = fsL.gas.p;
    number pLrL = pL/rhoL;
    number velxL = velL.x;
    number velyL = velL.y;
    number velzL = velL.z;
    number uL = fsL.gas.e;
    number aL = fsL.gas.a;
    number keL = 0.5*(velxL*velxL + velyL*velyL + velzL*velzL);
    number HL = uL + pLrL + keL;
    //
    number rhoR = fsR.gas.rho;
    number pR = fsR.gas.p;
    number pRrR = pR/rhoR;
    number velxR = velR.x;
    number velyR = velR.y;
    number velzR = velR.z;
    number uR = fsR.gas.e;
    number aR = fsR.gas.a;
    number keR = 0.5*(velxR*velxR + velyR*velyR + velzR*velzR);
    number HR = uR + pR/rhoR + keR;
    //
    // This is the main part of the flux calculator.
    //
    // Weighting parameters (eqn 32) for velocity splitting.
    number alphaL = 2.0*pLrL/(pLrL+pRrR);
    number alphaR = 2.0*pRrR/(pLrL+pRrR);
    // Common sound speed (eqn 33) and Mach numbers.
    number am = fmax(aL, aR);
    number ML = velxL/am;
    number MR = velxR/am;
    // Left state:
    // pressure splitting (eqn 34)
    // and velocity splitting (eqn 30)
    number pLplus, velxLplus;
    number dvelxL = 0.5 * (velxL + fabs(velxL));
    if (fabs(ML) <= 1.0) {
        pLplus = pL*(ML+1.0)*(ML+1.0)*(2.0-ML)*0.25;
        velxLplus = alphaL*((velxL+am)*(velxL+am)/(4.0*am) - dvelxL) + dvelxL;
    } else {
        pLplus = pL * dvelxL / velxL;
        velxLplus = dvelxL;
    }
    // Right state:
    // pressure splitting (eqn 34)
    // and velocity splitting (eqn 31)
    number pRminus, velxRminus;
    number dvelxR = 0.5*(velxR-fabs(velxR));
    if (fabs(MR) <= 1.0) {
        pRminus = pR*(MR-1.0)*(MR-1.0)*(2.0+MR)*0.25;
        velxRminus = alphaR*(-(velxR-am)*(velxR-am)/(4.0*am) - dvelxR) + dvelxR;
    } else {
        pRminus = pR * dvelxR / velxR;
        velxRminus = dvelxR;
    }
    // The mass flux. (eqn 29)
    number massL = velxLplus*rhoL;
    number massR = velxRminus*rhoR;
    number mass_half = massL+massR;
    // Pressure flux (eqn 34)
    number p_half = pLplus + pRminus;
    // Momentum flux: normal direction
    // Compute blending parameter s (eqn 37),
    // the momentum flux for AUSMV (eqn 21) and AUSMD (eqn 21)
    // and blend (eqn 36).
    number dp = pL - pR;
    constexpr number K_SWITCH = 10.0;
    dp = K_SWITCH * fabs(dp) / fmin(pL, pR);
    number s = 0.5 * fmin(1.0, dp);
    number rvel2_AUSMV = massL*velxL + massR*velxR;
    number rvel2_AUSMD = 0.5*(mass_half*(velxL+velxR) - fabs(mass_half)*(velxR-velxL));
    number rvel2_half = (0.5+s)*rvel2_AUSMV + (0.5-s)*rvel2_AUSMD;
    // Assemble components of the flux vector (eqn 36).
    f.F[CQI::mass] = mass_half;
    number vely = (mass_half >= 0.0) ? velyL : velyR;
    number velz = (mass_half >= 0.0) ? velzL : velzR;
    Vector3 momentum{rvel2_half+p_half, mass_half*vely, mass_half*velz};
    momentum.transform_to_global_frame(f.n, f.t1, f.t2);
    f.F[CQI::xMom] = momentum.x;
    f.F[CQI::yMom] = momentum.y;
    f.F[CQI::zMom] = momentum.z;
    number H = (mass_half >= 0.0) ? HL : HR;
    f.F[CQI::totEnergy] = mass_half*H;
    // When we introduce species, get the species flux lines from the Dlang code.
    // [TODO] PJ 2022-09-11
    return;
} // end ausmdv()

#endif
