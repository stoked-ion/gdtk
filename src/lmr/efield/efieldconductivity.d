/**
 * Models for gas conductivity. Perhaps this should be a gas model
 *
 * Author: Nick Gibbons
 * Version: 2021-05-24
 */

module lmr.efield.efieldconductivity;

import std.conv;
import std.format;
import std.math;
import std.stdio;

import gas.gas_model;
import gas.gas_state;
import gas.physical_constants;
import geom;
import nm.number;
import ntypes.complex;
import lmr.mass_diffusion;

interface ConductivityModel{
    @nogc number opCall(ref const(GasState) gs, const Vector3 pos, GasModel gm);
    /*
        Hall parameter beta = omega_ce/nu_e = e*Bz/(m_e*nu_e), signed by Bz, for the
        tensor (magnetised) conductivity. Models that have no physical electron
        collision frequency (test/constant correlations) return 0.0, i.e. the
        conductivity stays scalar and the Hall effect is off.
    */
    @nogc double hall_beta(ref const(GasState) gs, GasModel gm, double Bz);
}


class TestConductivity : ConductivityModel{
    this() {}
    final const number opCall(ref const(GasState) gs, const Vector3 pos, GasModel gm){
        double sigma = -1.0*exp(pos.x.re)*cos(pos.y.re);
        return to!number(sigma);
    }
    @nogc final double hall_beta(ref const(GasState) gs, GasModel gm, double Bz){ return 0.0; }
}

class ConstantConductivity : ConductivityModel{
/*
    Test with a just constant conductivity.
*/
    this() {}
    @nogc final number opCall(ref const(GasState) gs, const Vector3 pos, GasModel gm){
        return to!number(1.0);
    }
    @nogc final double hall_beta(ref const(GasState) gs, GasModel gm, double Bz){ return 0.0; }
}

class RaizerConductivity : ConductivityModel{
/*
    Test with the formula from: Y. P. Razier, Gas Discharge Physics (Springer-Verlag, 1991)
     - Valid for air, nitrogen and argon when weakly ionised.
*/
    this() {}
    @nogc final number opCall(ref const(GasState) gs, const Vector3 pos, GasModel gm){
        version(multi_T_gas) {
            double Tref;
            if (gm.n_modes == 0) {
                Tref = gs.T.re;
            } else {
                size_t iTe = gm.n_modes-1;
                Tref = gs.T_modes[iTe].re;
            }
        } else {
            double Tref = gs.T.re;
        }
        number sigma = 8300.0*exp(-36000.0/Tref);
        //debug{writefln(" gs: %s sigma: %e ", gs, sigma);}
        return sigma;
    }
    // The Raizer correlation gives sigma directly, with no underlying collision
    // frequency to build a Hall parameter from; Hall stays off with this model.
    @nogc final double hall_beta(ref const(GasState) gs, GasModel gm, double Bz){ return 0.0; }
}

class DiffusionConductivity : ConductivityModel{
/*
    Compute the electrical conductivity using an expression derived by NNG
    based on chapter 6 of Seshadri, 1925

     "Fundamentals of Plasma Physics"
     S. R. Seshadri
     American Elsivier Publishing Company, Inc. NU 10017

    @author: Nick Gibbons (09/12/21)
*/
    this(GasModel gm) {
        if (!gm.is_plasma) throw new Error("DiffusionConductivity model requires a GasModel with is_plasma=true");

        nsp = gm.n_species;
        number_density.length = nsp;
        Davg.length = nsp;
        // We lie to BinaryDiffusion that our is_plasma is false to prevent it from enforcing ambipolar diffusion
        bd = new BinaryDiffusion(nsp, false, gm.charge);
        electron_idx = gm.species_index("e-");
    }

    @nogc final number opCall(ref const(GasState) gs, const Vector3 pos, GasModel gm){
    /*
        We assume that the Einstein relations are valid for a multicomponent plasma.
          - Also assume no applied magnetic field and weak species gradients.
    */
        gm.massf2numden(gs, number_density);
        bd.computeAvgDiffCoeffs(gs, gm, Davg);

        number sigma = 0.0;
        foreach(i; 0 .. nsp){
            double Z = gm.charge[i];
            number n = (number_density[i] < to!number(0)) ? to!number(0.0) : number_density[i];
            number T = ((gm.n_modes > 0) && (i == electron_idx)) ? gs.T_modes[$-1] : gs.T;
            number D = Davg[i];
            sigma +=  Z*Z*D*n*elementary_charge*elementary_charge/Boltzmann_constant/T;
        }
        //debug{writefln(" gs: %s sigma: %e ", gs, sigma);}
        return sigma;
    }
    // No single electron collision frequency falls out of the multi-species
    // diffusion formulation; Hall stays off with this model.
    @nogc final double hall_beta(ref const(GasState) gs, GasModel gm, double Bz){ return 0.0; }
private:
    size_t nsp;
    number[] number_density;
    number[] Davg;
    BinaryDiffusion bd;
    int electron_idx;
}

class CoulombConductivity : ConductivityModel{
/*
    Two-temperature partially-ionised electrical conductivity:
        sigma = n_e e^2 / (m_e (nu_ea + nu_ei))
    with electron-neutral (Frost/Phelps fit) and electron-ion (Coulomb) collision
    frequencies, matching two_temperature_argon_kinetics.d. This is Coulomb-limited
    (Spitzer-like) at high ionisation, unlike the weakly-ionised Raizer/BOLSIG forms
    which omit electron-ion collisions. Evaluated in double precision (returns the
    real conductivity), like the Raizer model.

    @author: 2026
*/
    this(GasModel gm) {
        if (!gm.is_plasma) throw new Error("CoulombConductivity model requires a GasModel with is_plasma=true");
        nsp = gm.n_species;
        number_density.length = nsp;
        electron_idx = gm.species_index("e-");
    }

    @nogc final number opCall(ref const(GasState) gs, const Vector3 pos, GasModel gm){
        double n_e, nu;
        electron_collision_state(gs, gm, n_e, nu);
        double sigma = n_e*elementary_charge*elementary_charge/(_m_e*nu);
        number result = sigma;
        return result;
    }
    /*
        Hall parameter from the same electron collision frequency as the
        conductivity itself: beta = e*Bz/(m_e*nu_e), signed by Bz. Note this is
        sigma*Bz/(e*n_e), so the tensor conductivity built from (sigma, beta) is
        internally consistent.
    */
    @nogc final double hall_beta(ref const(GasState) gs, GasModel gm, double Bz){
        if (Bz == 0.0) return 0.0;
        double n_e, nu;
        electron_collision_state(gs, gm, n_e, nu);
        return elementary_charge*Bz/(_m_e*nu);
    }
    /*
        Complex-step-differentiable twins of opCall and hall_beta, used ONLY by the
        built-in MHD source (efieldsource.d) so that the Newton Jacobian sees how Joule
        heating depends on Te through sigma and beta. Same fits, clamps and constants
        as electron_collision_state, carried in `number`. Kept separate so the field
        solve and everything else stay on the real-valued path, bit for bit.
    */
    @nogc final number sigma_z(ref const(GasState) gs, GasModel gm){
        number n_e, nu;
        electron_collision_state_z(gs, gm, n_e, nu);
        return n_e*elementary_charge*elementary_charge/(_m_e*nu);
    }
    @nogc final number hall_beta_z(ref const(GasState) gs, GasModel gm, double Bz){
        number beta = 0.0;
        if (Bz == 0.0) return beta;
        number n_e, nu;
        electron_collision_state_z(gs, gm, n_e, nu);
        beta = elementary_charge*Bz/(_m_e*nu);
        return beta;
    }
private:
    @nogc void electron_collision_state_z(ref const(GasState) gs, GasModel gm, out number n_e, out number nu){
        gm.massf2numden(gs, number_density);
        n_e = 0.0;
        if (electron_idx >= 0) n_e = number_density[electron_idx];
        immutable size_t e_idx = (electron_idx >= 0) ? cast(size_t) electron_idx : size_t.max;
        number n_heavy = 0.0;
        foreach(i; 0 .. nsp){ if (i != e_idx) n_heavy += number_density[i]; }
        n_e = fmax(n_e, 1.0e10);
        number n_neutral = fmax(n_heavy - n_e, 1.0e16);
        number Te = (gm.n_modes > 0) ? gs.T_modes[$-1] : gs.T;
        Te = fmax(3000.0, fmin(Te, 500.0e3));
        number Q_ea;
        if (Te.re < 10.0e3) {
            Q_ea = 0.39 + Te*(-0.551e-4 + 0.595e-8*Te);
        } else {
            Q_ea = -0.35 + 0.775e-4*Te;
        }
        Q_ea *= 1.0e-20;
        number Q_ei = 1.95e-10/(Te*Te)*log(1.53e8*Te*Te*Te/(n_e/1.0e6));
        if (Q_ei.re < 0.0) Q_ei = 0.0;
        immutable double pi_d = PI;   // PI is `real`; keep the expression in Complex!double
        number v_th = sqrt(8.0*Boltzmann_constant*Te/(pi_d*_m_e));
        nu = fmax(n_neutral*v_th*Q_ea + n_e*v_th*Q_ei, 1.0e6);
    }
    @nogc void electron_collision_state(ref const(GasState) gs, GasModel gm, out double n_e, out double nu){
        gm.massf2numden(gs, number_density);
        n_e = (electron_idx >= 0) ? number_density[electron_idx].re : 0.0;
        // Sum heavy-particle densities (everything except electrons) and subtract the
        // ion density (= n_e by quasineutrality, single ionisation) to get neutrals.
        // This avoids depending on gm.charge, which some gas models (e.g. the reacting
        // argon model) do not populate.
        immutable size_t e_idx = (electron_idx >= 0) ? cast(size_t) electron_idx : size_t.max;
        double n_heavy = 0.0;
        foreach(i; 0 .. nsp){ if (i != e_idx) n_heavy += number_density[i].re; }
        n_e = fmax(n_e, 1.0e10);
        double n_neutral = fmax(n_heavy - n_e, 1.0e16);
        double Te = (gm.n_modes > 0) ? gs.T_modes[$-1].re : gs.T.re;
        Te = fmax(3000.0, fmin(Te, 500.0e3));
        // electron-neutral momentum-transfer cross-section [m^2] (Frost/Phelps fit)
        double Q_ea;
        if (Te < 10.0e3) {
            Q_ea = 0.39 + Te*(-0.551e-4 + 0.595e-8*Te);
        } else {
            Q_ea = -0.35 + 0.775e-4*Te;
        }
        Q_ea *= 1.0e-20;
        // electron-ion Coulomb cross-section [m^2] (n_e in cm^-3 inside the log)
        double Q_ei = 1.95e-10/(Te*Te)*log(1.53e8*Te*Te*Te/(n_e/1.0e6));
        if (Q_ei < 0.0) Q_ei = 0.0;
        double v_th = sqrt(8.0*Boltzmann_constant*Te/(PI*_m_e));
        nu = fmax(n_neutral*v_th*Q_ea + n_e*v_th*Q_ei, 1.0e6);
    }
    immutable double _m_e = 9.10938e-31; // electron mass [kg]
    size_t nsp;
    number[] number_density;
    int electron_idx;
}

class AirCoulombConductivity : ConductivityModel{
/*
    Partially-ionised electrical conductivity for high-temperature AIR:
        sigma = n_e e^2 / (m_e (nu_en + nu_ei))
    The electron-neutral momentum-transfer collision frequency is a species-weighted
    sum over N, O, N2, O2, NO using the cross-section fits of Gnoffo (1989) and
    Imamura (2018); the electron-ion term is the Coulomb (Spitzer-like) frequency.
    These are the SAME fits used in the air udf-source-terms.lua, so the field solver
    and the UDF Lorentz/Joule terms build a consistent (sigma, beta). This is the air
    analogue of CoulombConductivity (whose electron-neutral fit is argon-specific).

    @author: 2026
*/
    this(GasModel gm) {
        if (!gm.is_plasma) throw new Error("AirCoulombConductivity model requires a GasModel with is_plasma=true");
        nsp = gm.n_species;
        number_density.length = nsp;
        electron_idx = gm.species_index("e-");
        iN  = gm.species_index("N");
        iO  = gm.species_index("O");
        iN2 = gm.species_index("N2");
        iO2 = gm.species_index("O2");
        iNO = gm.species_index("NO");
        // Collect the indices of whichever singly-charged ions the model carries.
        foreach(nm; ["N2+", "O2+", "N+", "O+", "NO+"]){
            int idx = gm.species_index(nm);
            if (idx >= 0) ion_idx ~= idx;
        }
    }

    @nogc final number opCall(ref const(GasState) gs, const Vector3 pos, GasModel gm){
        double n_e, nu;
        electron_collision_state(gs, gm, n_e, nu);
        double sigma = n_e*elementary_charge*elementary_charge/(_m_e*nu);
        number result = sigma;
        return result;
    }
    /*
        Hall parameter from the same electron collision frequency as the conductivity,
        beta = e*Bz/(m_e*nu_e), signed by Bz, so the tensor conductivity built from
        (sigma, beta) is internally consistent.
    */
    @nogc final double hall_beta(ref const(GasState) gs, GasModel gm, double Bz){
        if (Bz == 0.0) return 0.0;
        double n_e, nu;
        electron_collision_state(gs, gm, n_e, nu);
        return elementary_charge*Bz/(_m_e*nu);
    }
private:
    @nogc void electron_collision_state(ref const(GasState) gs, GasModel gm, out double n_e, out double nu){
        gm.massf2numden(gs, number_density);
        n_e = (electron_idx >= 0) ? number_density[electron_idx].re : 0.0;
        n_e = fmax(n_e, 1.0e10);
        double nN  = (iN  >= 0) ? fmax(number_density[iN ].re, 0.0) : 0.0;
        double nO  = (iO  >= 0) ? fmax(number_density[iO ].re, 0.0) : 0.0;
        double nN2 = (iN2 >= 0) ? fmax(number_density[iN2].re, 0.0) : 0.0;
        double nO2 = (iO2 >= 0) ? fmax(number_density[iO2].re, 0.0) : 0.0;
        double nNO = (iNO >= 0) ? fmax(number_density[iNO].re, 0.0) : 0.0;
        double n_ions = 0.0;
        foreach(idx; ion_idx) n_ions += fmax(number_density[idx].re, 0.0);
        // Electron temperature is the last (electron/electronic) mode.
        double Te = (gm.n_modes > 0) ? gs.T_modes[$-1].re : gs.T.re;
        Te = fmax(3000.0, fmin(Te, 500.0e3));
        // electron-neutral momentum-transfer cross-sections [m^2] (Gnoffo 1989 / Imamura 2018)
        double Q_eN  = 5.0e-20;
        double Q_eO  = fmax(1.2e-20 + 1.7e-24*Te - 2.0e-28*Te*Te, 0.0);
        double Q_eN2 = fmax(7.5e-20 + 5.5e-24*Te - 1.0e-28*Te*Te, 0.0);
        double Q_eO2 = fmax(2.0e-20 + 6.0e-24*Te, 0.0);
        double Q_eNO = 1.0e-19;
        double v_th = sqrt(8.0*Boltzmann_constant*Te/(PI*_m_e));
        double nu_en = (4.0/3.0)*v_th*(Q_eN*nN + Q_eO*nO + Q_eN2*nN2 + Q_eO2*nO2 + Q_eNO*nNO);
        // electron-ion Coulomb collision frequency (same form as the air UDF)
        double nu_ei = 0.0;
        if (n_ions > 0.0) {
            double q2 = elementary_charge*elementary_charge;
            double a = q2/(12.0*PI*vacuum_permittivity*Boltzmann_constant*Te);
            double lnArg = 12.0*PI*pow(vacuum_permittivity*Boltzmann_constant/q2, 1.5)*sqrt(Te*Te*Te/n_e);
            double lnTerm = (lnArg > 1.0) ? log(lnArg) : 0.0;
            nu_ei = fmax((6.0*PI)*a*a*lnTerm*n_ions*v_th, 0.0);
        }
        nu = fmax(nu_en + nu_ei, 1.0e6);
    }
    immutable double _m_e = 9.10938e-31; // electron mass [kg]
    size_t nsp;
    number[] number_density;
    int electron_idx, iN, iO, iN2, iO2, iNO;
    int[] ion_idx;
}

ConductivityModel create_conductivity_model(string name, GasModel gm){
    ConductivityModel conductivity_model;
    switch (name) {
    case "test":
        conductivity_model = new TestConductivity();
        break;
    case "constant":
        conductivity_model = new ConstantConductivity();
        break;
    case "raizer":
        conductivity_model = new RaizerConductivity();
        break;
    case "diffusion":
        conductivity_model = new DiffusionConductivity(gm);
        break;
    case "coulomb":
        conductivity_model = new CoulombConductivity(gm);
        break;
    case "air_coulomb":
        conductivity_model = new AirCoulombConductivity(gm);
        break;
    case "none":
        break; //throw new Error("User has asked for solve_electric_field but failed to specify a conductivity model.");
    default:
        string errMsg = format("The conductivity model '%s' is not available.", name);
        throw new Error(errMsg);
    }
    return conductivity_model;
}
