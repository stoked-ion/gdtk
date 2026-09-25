/**
 * efieldsource.d
 *
 * Built-in low-magnetic-Reynolds-number MHD source: the Lorentz force J x B and the
 * Joule heating, formed from the SOLVED electric field and the applied magnetic field.
 *
 * This is what the X2 argon Lua UDFs compute. Doing it in D has one purpose a UDF cannot
 * serve: Lua sees only the real part of every quantity, so a UDF source enters the
 * complex-step Jacobian (preconditioner and Frechet products alike) as a constant.
 * Here the velocity is carried in `number` arithmetic, so the Jacobian sees the
 * magnetic braking dF_x/du_x = -sigma_P B^2 and the matching Joule terms. With the
 * Coulomb model, sigma and beta are also carried in `number` (mhd_source_differentiate_
 * sigma, default on), so the Jacobian sees how Joule heating depends on Te -- measured
 * to be the stiff term, where braking is not (~1e-3 per step at beta ~15). E is whatever
 * the field solve left in the cell, so this is the fixed-field part of the coupling.
 *
 * Generalised Ohm's law with the Hall tensor, identical to the UDF and to efield.d:
 *   J = sigma/(1+beta^2) [[1, -beta], [beta, 1]] (E + u x B),  B = Bz z-hat,
 *   E = -grad(phi)  (the cell stores +grad(phi)).
 * Sources:
 *   momentum:        J x B = (Jy Bz, -Jx Bz)
 *   total energy:    |J|^2/sigma + u.(J x B)
 *   electron energy: |J|^2/sigma   (the last energy mode; the Hall term does no work)
 *
 * Applies only to cells whose centre lies in the mhd_source_{x,y}{min,max} box, and
 * only once the field has been solved (before that the cell's field is NaN, and
 * J = sigma u x B with E = 0 would be a spurious short-circuit current).
 *
 * Author: 2026-09, gdtk-mhd branch.
 */

module lmr.efield.efieldsource;

import std.conv;
import std.math;

import nm.number;
import ntypes.complex;

import lmr.efield.efieldconductivity : CoulombConductivity;
import lmr.fluidfvcell;
import lmr.globalconfig;
import lmr.globaldata : SimState;

/// Smoothstep ramp on the step counter; 1 when no ramp is configured. The steady solver
/// passes t = -1, so a ramp on physical time would be zero throughout; ramp on steps.
@nogc double mhdSourceRampFactor()
{
    immutable int n = GlobalConfig.mhd_source_ramp_steps;
    if (n <= 0) return 1.0;
    double s = (SimState.step - GlobalConfig.mhd_source_ramp_start)/cast(double) n;
    if (s <= 0.0) return 0.0;
    if (s >= 1.0) return 1.0;
    return s*s*(3.0 - 2.0*s);
}

/// Add the Lorentz force and Joule heating to cell.Q. fs.gas.sigma must be current.
@nogc void addMHDSource(FluidFVCell cell, LocalConfig myConfig)
{
    if (!GlobalConfig.mhd_source) return;
    immutable double x = cell.pos[0].x.re, y = cell.pos[0].y.re;
    if (x < GlobalConfig.mhd_source_xmin || x > GlobalConfig.mhd_source_xmax) return;
    // The y-limits (typically a guard on the cell rows touching the electrodes) apply
    // only up to mhd_source_yguard_xmax, e.g. over a constant-area duct but not the
    // diverging nozzle downstream, whose walls lie outside [ymin, ymax].
    if (x <= GlobalConfig.mhd_source_yguard_xmax &&
        (y < GlobalConfig.mhd_source_ymin || y > GlobalConfig.mhd_source_ymax)) return;
    immutable double Exs = cell.electric_field[0], Eys = cell.electric_field[1];
    if (isNaN(Exs) || isNaN(Eys)) return;   // field not solved yet
    immutable double factor = mhdSourceRampFactor();
    if (factor == 0.0) return;

    auto gm = myConfig.gmodel;
    auto cqi = myConfig.cqi;
    immutable double Bz = appliedBzAt(x);
    immutable double Ex = -Exs, Ey = -Eys;   // physical E = -grad(phi)
    // Stage 2: with the Coulomb model, sigma and beta are recomputed here in `number`
    // (sigma_z/hall_beta_z) so the Jacobian sees d(Joule)/dTe. Otherwise, or with
    // mhd_source_differentiate_sigma = false, they are the real values (stage 1).
    immutable bool hall = GlobalConfig.electric_field_hall_effect && Bz != 0.0
        && (myConfig.conductivity_model !is null);
    auto coulomb = cast(CoulombConductivity) myConfig.conductivity_model;
    number sigma;
    number beta = 0.0;
    if (GlobalConfig.mhd_source_differentiate_sigma && coulomb !is null) {
        sigma = coulomb.sigma_z(cell.fs.gas, gm);
        if (hall) beta = coulomb.hall_beta_z(cell.fs.gas, gm, Bz);
    } else {
        sigma = cell.fs.gas.sigma;
        if (hall) beta = myConfig.conductivity_model.hall_beta(cell.fs.gas, gm, Bz);
    }
    if (sigma.re < 1.0e-10) sigma = 1.0e-10;
    number ux = cell.fs.vel.x, uy = cell.fs.vel.y;
    number s = sigma/(1.0 + beta*beta);
    number Jx = s*((Ex - beta*Ey) + Bz*(uy + beta*ux));
    number Jy = s*((Ey + beta*Ex) + Bz*(beta*uy - ux));
    number Fx = Jy*Bz, Fy = -Jx*Bz;
    number Qj = (Jx*Jx + Jy*Jy)/sigma;
    number W = ux*Fx + uy*Fy;

    cell.Q[cqi.xMom] += factor*Fx;
    cell.Q[cqi.yMom] += factor*Fy;
    cell.Q[cqi.totEnergy] += factor*(Qj + W);
    if (cqi.n_modes > 0) cell.Q[cqi.modes + cqi.n_modes - 1] += factor*Qj;
}
