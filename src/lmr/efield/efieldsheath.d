/**
 * Pluggable electrode-sheath current-voltage models for the low-Rm efield solver.
 * Phase 2 of the electrode-sheath BC (see SheathField in efieldbc.d).
 *
 * A SheathModel maps the sheath voltage dV = phi_plasma_edge - V_electrode to the
 * sheath current density J [A/m^2] flowing out of the plasma into the electrode, plus
 * its derivative dJ/d(dV) for the linearized Robin term that the (linear) field solve
 * assembles. Nonlinear laws are linearized locally around the current plasma-edge
 * potential each solve and converged by the Newton-Krylov outer loop.
 *
 * Author: (gdtk-mhd) 2026 -- mirrors the ConductivityModel pattern in efieldconductivity.d
 */
module lmr.efield.efieldsheath;

import std.conv;
import std.format;
import std.json;
import std.math;

import gas.gas_model;
import gas.gas_state;
import gas.physical_constants;
import util.json_helper;

interface SheathModel {
    // Sheath current density J [A/m^2] out of the plasma into the electrode, as a
    // function of the sheath voltage dV = phi_edge - V_electrode and the local plasma
    // edge state (gs, gm) -- the latter for emission/collection laws that need n_e, Te.
    double current(double dV, ref const(GasState) gs, GasModel gm);
    // Differential sheath conductance dJ/d(dV) [1/(Ohm.m^2)] for the linearized Robin term.
    double conductance(double dV, ref const(GasState) gs, GasModel gm);
}

// Linear sheath resistance: J = (dV - Vfall)/Rsheath, constant conductance 1/Rsheath.
// Reproduces the Phase 1 behaviour. Rsheath -> 0 = Dirichlet, Rsheath -> inf = open.
class LinearSheath : SheathModel {
    this(double Rsheath, double Vfall) {
        this.Rs = (Rsheath > 0.0) ? Rsheath : 1.0e-30;
        this.Vfall = Vfall;
    }
    final double current(double dV, ref const(GasState) gs, GasModel gm){ return (dV - Vfall)/Rs; }
    final double conductance(double dV, ref const(GasState) gs, GasModel gm){ return 1.0/Rs; }
private:
    double Rs, Vfall;
}

// Diode sheath: conducts (resistively, slope 1/Rsheath) only beyond the fall voltage
// |dV| > Vfall, and blocks below it. A small leak conductance is added everywhere to
// keep the matrix non-singular and the linearization smooth. Captures the switching
// nonlinearity of an electrode that needs a minimum over-voltage to pass current.
class DiodeSheath : SheathModel {
    this(double Rsheath, double Vfall, double leak) {
        this.Rs = (Rsheath > 0.0) ? Rsheath : 1.0e-30;
        this.Vfall = Vfall;
        this.leak = leak;
    }
    final double current(double dV, ref const(GasState) gs, GasModel gm){
        double J = leak*dV;
        if (dV >  Vfall) J += (dV - Vfall)/Rs;
        else if (dV < -Vfall) J += (dV + Vfall)/Rs;
        return J;
    }
    final double conductance(double dV, ref const(GasState) gs, GasModel gm){
        return (fabs(dV) > Vfall) ? (1.0/Rs + leak) : leak;
    }
private:
    double Rs, Vfall, leak;
}

// Child-Langmuir space-charge-limited sheath: J = K |dV|^1.5 (sign-preserving), plus a
// small leak for conditioning. K [A/(m^2 V^1.5)] is the lumped space-charge coefficient
// = (4/9) eps0 sqrt(2 q/m) / d_sheath^2 for the conducting carrier.
class ChildLangmuirSheath : SheathModel {
    this(double K, double leak, double dV_lin) { this.K = K; this.leak = leak; this.dV_lin = dV_lin; }
    final double current(double dV, ref const(GasState) gs, GasModel gm){
        double a = fabs(dV);
        return copysign(K*a*sqrt(a), dV) + leak*dV;
    }
    final double conductance(double dV, ref const(GasState) gs, GasModel gm){
        // The true differential conductance 1.5*K*sqrt(|dV|) vanishes at dV = 0 --
        // exactly where the FIRST field solve is seeded (the NaN guard linearizes
        // about phi = Velectrode). With it zero, nothing anchors the potential level
        // (the leak's ~1e-9 S row entries are the only gauge fixing), the system is
        // effectively pure-Neumann singular, and any rhs imbalance drives the level
        // to ~1e9 V and stalls the field GMRES. Floor the LINEARIZATION slope at its
        // |dV| = dV_lin value: the converged physics is untouched (the Robin fixed
        // point satisfies flux = S*J(dV) exactly, independent of the slope); only
        // the Newton-style damping near zero bias changes.
        return 1.5*K*sqrt(fmax(fabs(dV), dV_lin)) + leak;
    }
private:
    double K, leak, dV_lin;
}

// Electron-saturation (collector) sheath: the electrode can draw at most the random
// electron flux from the plasma edge, J_sat = (1/4) n_e q v_th,e with
// v_th,e = sqrt(8 k Te /(pi m_e)). Below saturation it is resistive (Rsheath, with the
// Vfall offset); the magnitude is capped at J_sat. A first cut at electrode collection
// physics; full thermionic (Richardson) emission from a hot cathode is a TODO.
class SaturationSheath : SheathModel {
    this(double Rsheath, double Vfall, double leak) {
        this.Rs = (Rsheath > 0.0) ? Rsheath : 1.0e-30;
        this.Vfall = Vfall;
        this.leak = leak;
    }
    private double jsat(ref const(GasState) gs, GasModel gm){
        int ie = gm.species_index("e-");
        if (ie < 0) return 1.0e30;                 // no electrons tracked -> effectively uncapped
        double n_e = Avogadro_number*gs.massf[ie].re*gs.rho.re/gm.mol_masses[ie];
        double Te  = (gm.n_modes > 0) ? gs.T_modes[gm.n_modes-1].re : gs.T.re;
        double m_e = gm.mol_masses[ie]/Avogadro_number;   // kg per electron
        double vth = sqrt(8.0*Boltzmann_constant*Te/(PI*m_e));
        return 0.25*n_e*elementary_charge*vth;
    }
    final double current(double dV, ref const(GasState) gs, GasModel gm){
        double Jlin = (dV - Vfall)/Rs;
        double Js = jsat(gs, gm);
        if (Jlin >  Js) return  Js + leak*dV;
        if (Jlin < -Js) return -Js + leak*dV;
        return Jlin + leak*dV;
    }
    final double conductance(double dV, ref const(GasState) gs, GasModel gm){
        double Jlin = (dV - Vfall)/Rs;
        double Js = jsat(gs, gm);
        return (fabs(Jlin) >= Js) ? leak : (1.0/Rs + leak);   // saturated -> only the leak slope
    }
private:
    double Rs, Vfall, leak;
}

// Factory: build a SheathModel from its name + the BC's JSON parameter table.
SheathModel create_sheath_model(string name, JSONValue j){
    double leak = getJSONdouble(j, "leak", 1.0e-6);
    switch (name) {
    case "linear":
        return new LinearSheath(getJSONdouble(j, "Rsheath", 1.0), getJSONdouble(j, "Vfall", 0.0));
    case "diode":
        return new DiodeSheath(getJSONdouble(j, "Rsheath", 1.0), getJSONdouble(j, "Vfall", 0.0), leak);
    case "child-langmuir":
        // dV_lin [V]: bias below which the Robin linearization slope is floored
        // (gauge anchoring for the first solve / zero-bias faces); see conductance().
        return new ChildLangmuirSheath(getJSONdouble(j, "K", 1.0e-3), leak,
                                       getJSONdouble(j, "dV_lin", 1.0));
    case "saturation":
        return new SaturationSheath(getJSONdouble(j, "Rsheath", 1.0), getJSONdouble(j, "Vfall", 0.0), leak);
    default:
        throw new Error(format("Unknown sheath_model '%s' (use linear|diode|child-langmuir|saturation).", name));
    }
}
