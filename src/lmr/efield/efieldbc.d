/**
 * Machinery for applying boundary conditions to electromagnetic fields
 *
 * Author: Nick Gibbons
 * Started: 2021-05-24
 */

module lmr.efield.efieldbc;

import std.conv;
import std.format;
import std.json;
import std.math;
import std.stdio;

import geom;
import nm.number;
import util.json_helper;

import gas.gas_model;

import lmr.bc.boundary_condition;
import lmr.bc.ghost_cell_effect.full_face_copy;
import lmr.efield.efieldcircuit;
import lmr.globaldata : SimState;
import lmr.efield.efieldconductivity;
import lmr.efield.efieldsheath;
import lmr.globalconfig;
import lmr.fluidfvcell;
import lmr.fvinterface;

interface FieldBC {
    bool isShared() const;
    Vector3 other_pos(const FVInterface face);
    int other_id(const FVInterface face);
    double phif(const FVInterface face);
    double lhs_direct_component(double fac, const FVInterface face);
    double lhs_other_component(double fac, const FVInterface face);
    double rhs_direct_component(double sign, double fac, const FVInterface face);
    double rhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface);
    double lhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface);
    double compute_current(const double sign, const FVInterface face, const FluidFVCell cell);
}

class ZeroNormalGradient : FieldBC {
    this() {}

    final bool isShared() const { return false; }
    final Vector3 other_pos(const FVInterface face) {return face.pos;}
    final int other_id(const FVInterface face) {return -1;}
    final double phif(const FVInterface face) { return 0.0;}
    final double lhs_direct_component(double fac, const FVInterface face){ return 0.0;}
    final double lhs_other_component(double fac, const FVInterface face){ return 0.0;}
    final double rhs_direct_component(double sign, double fac, const FVInterface face){ return 0.0;}
    final double rhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){ return 0.0; }
    final double lhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){ return 0.0; }
    final double compute_current(const double sign, const FVInterface face, const FluidFVCell cell){ return 0.0; }

    override string toString() const
    {
        return "ZeroGradient()";
    }
}

class FixedField : FieldBC {
    this(double value) {
        this.value = value;
    }

    final bool isShared() const { return false; }
    final Vector3 other_pos(const FVInterface face) {return face.pos;}
    final int other_id(const FVInterface face) {return -1;}
    final double phif(const FVInterface face) { return value;}
    final double lhs_direct_component(double fac, const FVInterface face){ return -1.0*face.length.re*fac*face.fs.gas.sigma.re;}
    final double lhs_other_component(double fac, const FVInterface face){ return 0.0;}
    final double rhs_direct_component(double sign, double fac, const FVInterface face){ return face.length.re*fac*face.fs.gas.sigma.re*value;}
    final double rhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){
        return (facx*fdx + facy*fdy)/D*value;
    }
    final double lhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface) { return 0.0; }

    final double compute_current(const double sign, const FVInterface face, const FluidFVCell cell){
        double S = face.length.re;
        double d = distance_between(face.pos, cell.pos[0]);
        double phigrad = (value - cell.electric_potential)/d; // This implicitly points out of the domain.
        double sigma = face.fs.gas.sigma.re;
        double I = sigma*phigrad*S; // convert from current density vector j to current I
        return I;
    }
    override string toString() const
    {
        char[] repr;
        repr ~= "FixedValue(";
        repr ~= "value=" ~ to!string(value);
        repr ~= ")";
        return to!string(repr);
    }
private:
    double value;
}

class SheathField : FieldBC {
/*
    Electrode sheath boundary condition (Phase 2: pluggable SheathModel).

    Motivation: the cold electrode face has Raizer sigma = 8300*exp(-36000/Te) ~ 0,
    because the no-slip fixed-T wall forces Te = Twall (~300 K) at the face. A FixedField
    (Dirichlet) BC's matrix weight is proportional to that face sigma, so it collapses to
    ~0 and the applied electrode voltage has no effect -- the device sits at open circuit
    regardless of voltage. This BC instead inserts a physical sheath impedance in series
    between the electrode metal (potential Velectrode) and the plasma edge (boundary cell).

    The sheath current-voltage law J(dV), dV = phi_cell - Velectrode, is a pluggable
    SheathModel (efieldsheath.d: linear | diode | child-langmuir | saturation). The Robin
    term it produces is assembled directly in efield.d via linearized_robin(): the model
    is linearized locally around the current plasma-edge potential each solve and the
    nonlinearity is converged by the Newton-Krylov outer loop. The matrix weight is the
    sheath's differential conductance dJ/d(dV), NOT the collapsed cold-face gas sigma --
    so the applied voltage couples to the bulk. The gas-conduction stencil at this face
    is suppressed (treated like ZeroNormalGradient in efield.d's assembly). For a linear
    model, Rsheath -> 0 recovers a hard Dirichlet (FixedField); Rsheath -> inf recovers
    open circuit (insulator).
*/
    this(double Velectrode, SheathModel model,
         double segment_pitch=0.0, double segment_fill=1.0, double segment_x0=0.0,
         double Ex_applied=0.0, double Ex_quad=0.0, double Ex_cube=0.0,
         double Ex_ramp_steps=0.0, double Ex_ramp_start=0.0,
         double segment_x1=0.0) {
        this.Velectrode = Velectrode;
        this.model = model;
        this.segment_pitch = segment_pitch;
        this.segment_fill = segment_fill;
        this.segment_x0 = segment_x0;
        this.Ex_applied = Ex_applied;
        this.Ex_quad = Ex_quad;
        this.Ex_cube = Ex_cube;
        this.Ex_ramp_steps = Ex_ramp_steps;
        this.Ex_ramp_start = Ex_ramp_start;
        this.segment_x1 = segment_x1;
    }

    /*
        Fraction of the axial tilt currently applied.

        A tilt is a large perturbation: at condition 6 the tilt matching tan(theta) = beta is
        1838 V across a duct driven at 400 V. Imposing it in one step destroys the solve even
        when restarting from a fully converged field and running first order -- measured, the
        run dies within ten steps. Ramping it over a few hundred Newton steps is the same
        remedy the deferred UDF source terms need, and for the same reason.

        The steady solver passes SimState.time = -1, so the ramp is on SimState.step. A
        smoothstep is used rather than a linear ramp: a linear one puts a kink in the
        residual at each end, which a Newton solver feels.

        Ex_ramp_steps <= 0 (the default) means no ramp, and the expression below reduces
        exactly to the previous behaviour.
    */
    @nogc final double Ex_ramp_factor() const {
        if (!(Ex_ramp_steps > 0.0)) return 1.0;
        double t = (cast(double) SimState.step - Ex_ramp_start)/Ex_ramp_steps;
        if (t <= 0.0) return 0.0;
        if (t >= 1.0) return 1.0;
        return t*t*(3.0 - 2.0*t);
    }

    /*
        Diagonal-mode electrode: the electrode metal potential follows a polynomial profile
        along the channel axis,
            V(x) = Velectrode + Ex_applied*dx + Ex_quad*dx^2 + Ex_cube*dx^3,   dx = x - segment_x0,
        so the imposed axial field is  -dV/dx = -(Ex_applied + 2 Ex_quad dx + 3 Ex_cube dx^2).
        With the SAME profile on the anode and the cathode wall, the equipotentials are tilted
        by the diagonal angle theta (optimally tan(theta) = beta), recovering close to the full
        conductivity at high Hall parameter where flat Faraday electrodes give only
        sigma/(1+beta^2). A pure linear ramp (Ex_quad = Ex_cube = 0) tilts at a single angle;
        the quadratic/cubic terms let the tilt track a spatially-varying Hall field beta*E'_y(x)
        (which can swing by an order of magnitude along a strongly-inhomogeneous channel) -- a
        constant Ex_applied can only match it at one station. Ex_applied = 0 (the default) is a
        flat electrode = Faraday mode.
    */
    @nogc final double Velectrode_at(const FVInterface face) const {
        double dx = face.pos.x.re - segment_x0;
        return Velectrode + Ex_ramp_factor()*dx*(Ex_applied + dx*(Ex_quad + dx*Ex_cube));
    }

    /*
        Segmented electrodes: with the Hall effect on, a continuous conductor
        short-circuits the axial (Hall) field along the wall and kills the Faraday
        current; real MHD channels break the electrode into segments separated by
        insulator strips. The wall is described periodically: starting from
        segment_x0, each pitch of length segment_pitch is electrode for the first
        segment_fill fraction and insulator for the rest. segment_pitch <= 0 (the
        default) means a continuous electrode. Insulator faces get no sheath Robin
        term and carry no current (J.n = 0), exactly like ZeroNormalGradient in the
        assembly's treatment of this BC.
    */
    /*
        Axial extent of the segmented region. The periodic tiling above runs the whole
        length of the wall, which is right when the electrodes fill the duct but wrong
        for a channel longer than the magnet: a segment landing outside the field
        shorts the plasma while generating no EMF. segment_x1 > segment_x0 restricts
        the metal to [segment_x0, segment_x1]; leaving it at the default 0.0 keeps the
        pre-existing unbounded behaviour exactly.
    */
    @nogc final bool in_segment_window(const FVInterface face) const {
        if (!(segment_x1 > segment_x0)) return true;
        double xf = face.pos.x.re;
        return (xf >= segment_x0) && (xf <= segment_x1);
    }

    @nogc final bool is_electrode(const FVInterface face) const {
        if (!in_segment_window(face)) return false;
        if (segment_pitch <= 0.0) return true;
        double s = (face.pos.x.re - segment_x0) % segment_pitch;
        if (s < 0.0) s += segment_pitch;
        return s < segment_fill*segment_pitch;
    }

    final bool isShared() const { return false; }
    final Vector3 other_pos(const FVInterface face) {return face.pos;}
    final int other_id(const FVInterface face) {return -1;}
    final double phif(const FVInterface face) { return 0.0; } // gas gradient is ZNG-like here
    // The sheath Robin term is assembled in efield.d via linearized_robin(); these
    // FieldBC interface stubs are unused for SheathField (the assembly handles it).
    final double lhs_direct_component(double fac, const FVInterface face){ return 0.0; }
    final double lhs_other_component(double fac, const FVInterface face){ return 0.0; }
    final double rhs_direct_component(double sign, double fac, const FVInterface face){ return 0.0; }
    final double rhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){ return 0.0; }
    final double lhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){ return 0.0; }
    final double compute_current(const double sign, const FVInterface face, const FluidFVCell cell){
        if (!is_electrode(face)) return 0.0; // insulator strip between segments
        double S = face.length.re;
        double dV = cell.electric_potential.re - Velectrode_at(face);
        return model.current(dV, face.fs.gas, GlobalConfig.gmodel_master)*S; // sheath current out into electrode
    }

    // Linearized Robin contribution for the boundary cell's charge balance, assembled in
    // efield.d. dV = phi_cell - Velectrode; linearize J(dV) about the current phi_cell:
    //   I_into_cell = -S*J(dV) ~= -S*Jp*phi_cell + S*(Jp*phi_cell - J0)
    // giving A[diag] += a_diag, b[k] += b_rhs with:
    void linearized_robin(const FVInterface face, double phi_cell, GasModel gm, out double a_diag, out double b_rhs){
        double S = face.length.re;
        // NaN guard: cell.electric_potential is NaN before the first solve. A nonlinear
        // model linearized about NaN gives a NaN matrix that never recovers (the linear
        // model is immune because phi_cell cancels). Seed the first linearization with
        // dV0 = 0 (phi_cell = Velectrode) so it starts finite; later solves use the real phi.
        if (phi_cell != phi_cell) phi_cell = Velectrode_at(face);
        double dV0 = phi_cell - Velectrode_at(face);
        double J0 = model.current(dV0, face.fs.gas, gm);
        double Jp = model.conductance(dV0, face.fs.gas, gm);
        a_diag = -S*Jp;
        b_rhs  = S*(J0 - Jp*phi_cell);
    }
    override string toString() const {
        return format("SheathField(Velectrode=%g, segment_pitch=%g, segment_fill=%g, Ex_applied=%g,"
                      ~ " Ex_quad=%g, Ex_cube=%g, Ex_ramp_steps=%g, Ex_ramp_start=%g)",
                      Velectrode, segment_pitch, segment_fill, Ex_applied, Ex_quad, Ex_cube,
                      Ex_ramp_steps, Ex_ramp_start);
    }
private:
    double Velectrode;
    SheathModel model;
    double segment_pitch, segment_fill, segment_x0, segment_x1;
    double Ex_applied, Ex_quad, Ex_cube;
    double Ex_ramp_steps, Ex_ramp_start;
}

class CircuitElectrode : FieldBC {
/*
    Electrode wired into an EXTERNAL CIRCUIT (Path 1).

    Identical to SheathField in every respect -- same pluggable SheathModel, same
    segmented-electrode geometry, same linearized Robin term -- with exactly one
    difference: the electrode metal potential is NOT a constant from the Lua input
    file. It is the unknown q_m of circuit node `node_id`, solved simultaneously
    with the field.

    Why that difference matters: a fixed Velectrode models an electrode wired to its
    own ideal, zero-impedance supply. That is correct for Faraday (independent pairs)
    and for diagonal (a resistor ladder whose taps have negligible impedance), which
    is why those connections already work with SheathField and should NOT be migrated
    to this BC. It is wrong for any connection where electrodes are wired TO EACH
    OTHER -- a Hall short, or segments sharing a ballast network -- because prescribing
    both terminals' potentials supplies no equation limiting the current that flows
    between them. See efieldcircuit.d's header and the Path 1 plan, Sec. 1.2.

    Two electrode groups wired together simply name the SAME node id; that shared id
    IS the short, and the Kirchhoff row for that node is what limits the current.

    The assembly in efield.d branches on this type and routes the sheath linearization
    through sheathFaceStamp (efieldcircuit.d), which is the single place the augmented
    system's sign convention lives; the augmented system is then solved by the
    Woodbury/Schur path in the same module. Verified two ways: the stamps reduce to
    SheathField's own (a_diag, b_rhs) when the node is frozen (unit tests in
    efieldcircuit.d), and a whole case re-expressed with one node per electrode and
    R -> 0 reproduces the SheathField result to 0.05% in F_x and total current.
*/
    this(ExternalCircuit circuit, int node_id, SheathModel model,
         double segment_pitch=0.0, double segment_fill=1.0, double segment_x0=0.0,
         double segment_x1=0.0) {
        this.circuit = circuit;
        this.node_id = node_id;
        this.model = model;
        this.segment_pitch = segment_pitch;
        this.segment_fill = segment_fill;
        this.segment_x0 = segment_x0;
        this.segment_x1 = segment_x1;
    }

    // The electrode metal potential: the circuit node's CURRENT estimate. Before the
    // first solve this is the node's nominal_voltage (see ExternalCircuit.addNode),
    // which keeps the first sheath linearization finite -- the same role the constant
    // Velectrode plays in SheathField's NaN guard.
    final double Velectrode_at(const FVInterface face) const {
        return circuit.q_of(node_id);
    }

    // Identical to SheathField.is_electrode -- segmented electrodes leave insulator
    // strips between segments so a continuous conductor cannot short the axial Hall
    // field along the wall.
    /*
        Axial extent of the segmented region. The periodic tiling above runs the whole
        length of the wall, which is right when the electrodes fill the duct but wrong
        for a channel longer than the magnet: a segment landing outside the field
        shorts the plasma while generating no EMF. segment_x1 > segment_x0 restricts
        the metal to [segment_x0, segment_x1]; leaving it at the default 0.0 keeps the
        pre-existing unbounded behaviour exactly.
    */
    @nogc final bool in_segment_window(const FVInterface face) const {
        if (!(segment_x1 > segment_x0)) return true;
        double xf = face.pos.x.re;
        return (xf >= segment_x0) && (xf <= segment_x1);
    }

    @nogc final bool is_electrode(const FVInterface face) const {
        if (!in_segment_window(face)) return false;
        if (segment_pitch <= 0.0) return true;
        double s = (face.pos.x.re - segment_x0) % segment_pitch;
        if (s < 0.0) s += segment_pitch;
        return s < segment_fill*segment_pitch;
    }

    final int nodeId() const { return node_id; }

    final bool isShared() const { return false; }
    final Vector3 other_pos(const FVInterface face) {return face.pos;}
    final int other_id(const FVInterface face) {return -1;}
    final double phif(const FVInterface face) { return 0.0; } // gas gradient is ZNG-like here
    final double lhs_direct_component(double fac, const FVInterface face){ return 0.0; }
    final double lhs_other_component(double fac, const FVInterface face){ return 0.0; }
    final double rhs_direct_component(double sign, double fac, const FVInterface face){ return 0.0; }
    final double rhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){ return 0.0; }
    final double lhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){ return 0.0; }
    final double compute_current(const double sign, const FVInterface face, const FluidFVCell cell){
        if (!is_electrode(face)) return 0.0; // insulator strip between segments
        double S = face.length.re;
        double dV = cell.electric_potential.re - Velectrode_at(face);
        return model.current(dV, face.fs.gas, GlobalConfig.gmodel_master)*S;
    }

    /*
        Sheath linearization, split so the field assembly can route each piece to the
        right place in the augmented system. Compare SheathField.linearized_robin,
        which folds the electrode-potential dependence entirely into b_rhs because
        there Velectrode is a constant.

        Linearizing J about the frozen pair (phi_cell*, q_m*) -- both held at their
        prior iterate, a Picard step, exactly as SheathField already freezes Jp with
        respect to phi_cell:

            J(phi_cell, q_m) ~= J0 + Jp*(phi_cell - phi_cell*) - Jp*(q_m - q_m*)

        so the face's contribution to the cell's charge balance, -S*J, matched against
        a_diag*phi_cell + u_coeff*q_m - b_rhs gives

            a_diag  = -S*Jp        (unchanged from SheathField)
            u_coeff = +S*Jp        (OPPOSITE sign to a_diag)
            b_rhs   =  S*(J0 - Jp*phi_cell* + Jp*q_m*)

        The +S*Jp*q_m* term in b_rhs is easy to omit; without it the row does not
        reduce to SheathField's when q_m is held fixed, which is exactly what the
        Phase 2 R->0 degeneracy test checks. Do NOT differentiate Jp with respect to
        q_m -- that would be inconsistent with how phi_cell's nonlinearity is already
        handled here and in SheathField.
    */
    // Expose the sheath model's frozen-point current and differential conductance so
    // the field assembly can feed them to sheathFaceStamp (efieldcircuit.d), which is
    // the single place the augmented-system sign convention lives.
    void sheathCurrentAndConductance(const FVInterface face, double phi_star, GasModel gm,
                                     out double J0, out double Jp) {
        double dV0 = phi_star - Velectrode_at(face);
        J0 = model.current(dV0, face.fs.gas, gm);
        Jp = model.conductance(dV0, face.fs.gas, gm);
    }

    void linearized_robin_circuit(const FVInterface face, double phi_cell, GasModel gm,
                                  out double a_diag, out double u_coeff, out double b_rhs){
        double S = face.length.re;
        double q_m = Velectrode_at(face);
        if (phi_cell != phi_cell) phi_cell = q_m; // NaN guard, as in SheathField
        double dV0 = phi_cell - q_m;
        double J0 = model.current(dV0, face.fs.gas, gm);
        double Jp = model.conductance(dV0, face.fs.gas, gm);
        a_diag  = -S*Jp;
        u_coeff =  S*Jp;
        b_rhs   =  S*(J0 - Jp*phi_cell + Jp*q_m);
    }

    override string toString() const {
        return format("CircuitElectrode(node=%d, segment_pitch=%g, segment_fill=%g)",
                      node_id, segment_pitch, segment_fill);
    }
private:
    ExternalCircuit circuit;
    int node_id;
    SheathModel model;
    double segment_pitch, segment_fill, segment_x0, segment_x1;
}

class MixedField : FieldBC {
    this(double differential, double xinsulator, double xcollector) {
        this.nose = new FixedField(1.0);
        this.insulator = new ZeroNormalGradient();
        this.collector = new FixedField(1.0+differential);
        this.xinsulator = xinsulator;
        this.xcollector = xcollector;
    }

    final bool isShared() const { return false; }

    final Vector3 other_pos(const FVInterface face) {return face.pos;}

    final int other_id(const FVInterface face) {return -1;}

    final double phif(const FVInterface face) {
        if (face.pos.x<xinsulator){
            return nose.phif(face);
        } else if (face.pos.x<xcollector) {
            return insulator.phif(face);
        } else {
            return collector.phif(face);
        }
    }

    final double lhs_direct_component(double fac, const FVInterface face){
        if (face.pos.x<xinsulator){
            return nose.lhs_direct_component(fac, face);
        } else if (face.pos.x<xcollector) {
            return insulator.lhs_direct_component(fac, face);
        } else {
            return collector.lhs_direct_component(fac, face);
        }
    }

    final double lhs_other_component(double fac, const FVInterface face){
        if (face.pos.x<xinsulator){
            return nose.lhs_other_component(fac, face);
        } else if (face.pos.x<xcollector) {
            return insulator.lhs_other_component(fac, face);
        } else {
            return collector.lhs_other_component(fac, face);
        }
    }

    final double rhs_direct_component(double sign, double fac, const FVInterface face){
        if (face.pos.x<xinsulator){
            return nose.rhs_direct_component(sign, fac, face);
        } else if (face.pos.x<xcollector) {
            return insulator.rhs_direct_component(sign, fac, face);
        } else {
            return collector.rhs_direct_component(sign, fac, face);
        }
    }

    final double rhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){
        if (jface.pos.x<xinsulator){
            return nose.rhs_stencil_component(D, facx, facy, fdx, fdy, jface);
        } else if (jface.pos.x<xcollector) {
            return insulator.rhs_stencil_component(D, facx, facy, fdx, fdy, jface);
        } else {
            return collector.rhs_stencil_component(D, facx, facy, fdx, fdy, jface);
        }
    }

    final double lhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){
        if (jface.pos.x<xinsulator){
            return nose.lhs_stencil_component(D, facx, facy, fdx, fdy, jface);
        } else if (jface.pos.x<xcollector) {
            return insulator.lhs_stencil_component(D, facx, facy, fdx, fdy, jface);
        } else {
            return collector.lhs_stencil_component(D, facx, facy, fdx, fdy, jface);
        }
    }

    final double compute_current(const double sign, const FVInterface face, const FluidFVCell cell){
        double I;
        if (face.pos.x<xinsulator){
            I = nose.compute_current(sign, face, cell);
        } else if (face.pos.x<xcollector) {
            I = insulator.compute_current(sign, face, cell);
        } else {
            I = collector.compute_current(sign, face, cell);
        }
        return I;
    }
private:
    double xinsulator, xcollector;
    FixedField nose, collector;
    ZeroNormalGradient insulator;
}

class FixedField_Test : FieldBC {
    this() {}

    final bool isShared() const { return false; }
    final Vector3 other_pos(const FVInterface face) {return face.pos;}
    final int other_id(const FVInterface face) {return -1;}
    final double phif(const FVInterface face) { return exp(face.pos.x.re)*sin(face.pos.y.re);}
    final double lhs_direct_component(double fac, const FVInterface face){ return -1.0*face.length.re*fac*face.fs.gas.sigma.re;}
    final double lhs_other_component(double fac, const FVInterface face){ return 0.0;}
    final double rhs_direct_component(double sign, double fac, const FVInterface face){ return face.length.re*fac*face.fs.gas.sigma.re*phif(face);}
    final double rhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){
        return (facx*fdx + facy*fdy)/D*phif(jface);
    }
    final double lhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface) { return 0.0; }

    final double compute_current(const double sign, const FVInterface face, const FluidFVCell cell){
        double S = face.length.re;
        double d = distance_between(face.pos, cell.pos[0]);
        double phi = test_field(face.pos.x.re, face.pos.y.re);
        double phigrad = (phi - cell.electric_potential)/d; // This implicitly points out of the domain.
        double sigma = face.fs.gas.sigma.re;
        double I = sigma*phigrad*S;
        return I;
    }
    double test_field(double x, double y){
        return exp(x)*sin(y);
    }
    void test_field_gradient(double x, double y, ref double dphidx, ref double dphidy){
        dphidx = exp(x)*sin(y);
        dphidy = exp(x)*cos(y);
        return;
    }
}

class FixedGradient_Test : FieldBC {
    this() {
    }

    final bool isShared() const { return false; }
    final Vector3 other_pos(const FVInterface face) {return face.pos;}
    final int other_id(const FVInterface face) {return -1;}
    final double phif(const FVInterface face) { return exp(face.pos.x.re)*sin(face.pos.y.re);}
    final double lhs_direct_component(double fac, const FVInterface face){ return 0.0;}
    final double lhs_other_component(double fac, const FVInterface face){ return 0.0;}
    final double rhs_direct_component(double sign, double fac, const FVInterface face){
        double S = face.length.re;
        double sigma = face.fs.gas.sigma.re;
        Vector3 phigrad = test_field_gradient(face.pos.x.re, face.pos.y.re);

        number phigrad_dot_n = phigrad.dot(face.n);
        return sign*phigrad_dot_n.re*S*sigma;
    }
    final double rhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){ return 0.0; }
    final double lhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface) { return 0.0; }

    final double compute_current(const double sign, const FVInterface face, const FluidFVCell cell){
        double S = face.length.re;
        Vector3 phigrad = test_field_gradient(face.pos.x.re, face.pos.y.re);
        number phigrad_dot_n = sign*phigrad.dot(face.n); // TODO: Should this be negative sign?
        double sigma = face.fs.gas.sigma.re;
        double I = sigma*phigrad_dot_n.re*S;
        return I;
    }
private:
    Vector3 test_field_gradient(double x, double y){
        Vector3 phigrad = Vector3(exp(x)*sin(y), exp(x)*cos(y), 0.0);
        return phigrad;
    }
}

class SharedField : FieldBC {
    int other_blk_id;
    int other_block_offset;
    int[] other_cell_ids;
    bool[] other_cell_lefts;

    this(const BoundaryCondition bc, const int[] block_offsets) {
        GhostCellFullFaceCopy gc;
        foreach(action; bc.preReconAction){
            gc = cast(GhostCellFullFaceCopy) action;
            if (gc !is null){
                break;
            }
        }
        if (gc is null){
            throw new Error("Boundary is missing full_face_copy. Possibly check all outer boundaries are assigned.");
        }

        // We keep our own copies of data related to the shared boundary
        other_blk_id = gc.other_blk.id;
        other_block_offset = block_offsets[other_blk_id];

        // Since the arrays inside the GhostCellFullFaceCopy are not in the same order as the boundary faces
        // we have to do some work to organise our own mapping array, of boundary faces to shared cell ids.
        other_cell_ids.length = bc.faces.length;
        other_cell_lefts.length = bc.faces.length;
        foreach(i, f; bc.faces){
            foreach(j, c; gc.ghost_cells){
                if (f.right_cell==c) {
                    other_cell_lefts[i] = false;
                    other_cell_ids[i] = to!int(gc.mapped_cell_ids[j]);
                    break;
                }
                if (f.left_cell==c) {
                    other_cell_lefts[i] = true;
                    other_cell_ids[i] = to!int(gc.mapped_cell_ids[j]);
                    break;
                }
            }
        }
        return;
    }


    final bool isShared() const { return true; }
    final Vector3 other_pos(const FVInterface face) {return (other_cell_lefts[face.i_bndry]) ? face.left_cell.pos[0] : face.right_cell.pos[0];}
    final int other_id(const FVInterface face) {return other_cell_ids[face.i_bndry] + other_block_offset;}
    final double phif(const FVInterface face) { return (other_cell_lefts[face.i_bndry]) ? face.left_cell.electric_potential : face.right_cell.electric_potential;}
    double lhs_direct_component(double fac, const FVInterface face){ return -1.0*face.length.re*fac*face.fs.gas.sigma.re; }
    double lhs_other_component(double fac, const FVInterface face){ return 1.0*face.length.re*fac*face.fs.gas.sigma.re; }
    double rhs_direct_component(double sign, double fac, const FVInterface face){ return 0.0; }
    final double rhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){ return 0.0; }
    final double lhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){
        return (facx*fdx + facy*fdy)/D;
    }
    final double compute_current(const double sign, const FVInterface face, const FluidFVCell cell){
        return 0.0;
    }
}

version(mpi_parallel){
class MPISharedField : FieldBC {
    static int nExtraCells=0;
    static MPISharedField[] instances;
    int other_blk_rank;
    int other_blk_face;
    int other_blk_id;
    int[] other_cell_ids;

    this(const BoundaryCondition bc, const int ncells) {

        GhostCellFullFaceCopy gc;
        foreach(action; bc.preReconAction){
            gc = cast(GhostCellFullFaceCopy) action;
            if (gc !is null){
                break;
            }
        }
        if (gc is null){
            throw new Error("Boundary is missing full_face_copy. Possibly check all outer boundaries are assigned.");
        }

        // In MPI mode we assume that each block has its own process. We need to know:
        // 1.) Where in the extra cell array each othercell goes, for setting Ai in the matrix
        // 2.) We need to know where in the other block that other cell goes, so that the sending
        //     process can send it to us.
        // This extra cell array will be order of faces, but also in order of the SharedBCs being initialised.
        other_blk_rank = gc.other_blk_rank;
        other_blk_id = gc.other_blk.id;
        other_blk_face = gc.other_face;

        // Since the arrays inside the GhostCellFullFaceCopy are not in the same order as the boundary faces
        // we have to do some work to organise our own mapping array, of boundary faces to shared cell ids.
        other_cell_lefts.length = bc.faces.length;
        other_cell_ids.length = bc.faces.length;
        external_cell_idxs.length = bc.faces.length;
        my_offset = nExtraCells + ncells;

        foreach(i, f; bc.faces){
            foreach(j, c; gc.ghost_cells){
                if (f.right_cell==c) {
                    other_cell_lefts[i] = false;
                    other_cell_ids[i] = to!int(gc.mapped_cell_ids[j]);
                    external_cell_idxs[i] = to!int(i) + my_offset;
                    break;
                }
                if (f.left_cell==c) {
                    other_cell_lefts[i] = true;
                    other_cell_ids[i] = to!int(gc.mapped_cell_ids[j]);
                    external_cell_idxs[i] = to!int(i) + my_offset;
                    break;
                }
            }
        }
        nExtraCells += bc.faces.length;
        instances ~= this;
        return;
    }

    final bool isShared() const { return true; }
    final Vector3 other_pos(const FVInterface face) {return (other_cell_lefts[face.i_bndry]) ? face.left_cell.pos[0] : face.right_cell.pos[0];}
    final int other_id(const FVInterface face) {return external_cell_idxs[face.i_bndry];}
    final double phif(const FVInterface face) { return (other_cell_lefts[face.i_bndry]) ? face.left_cell.electric_potential : face.right_cell.electric_potential;}
    double lhs_direct_component(double fac, const FVInterface face){ return -1.0*face.length.re*fac*face.fs.gas.sigma.re; }
    double lhs_other_component(double fac, const FVInterface face){ return 1.0*face.length.re*fac*face.fs.gas.sigma.re; }
    double rhs_direct_component(double sign, double fac, const FVInterface face){ return 0.0; }
    final double rhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){ return 0.0; }
    final double lhs_stencil_component(double D, double facx, double facy, double fdx, double fdy, FVInterface jface){
        return (facx*fdx + facy*fdy)/D;
    }
    final double compute_current(const double sign, const FVInterface face, const FluidFVCell cell){
        return 0.0;
    }

private:
    bool[] other_cell_lefts;
    int[] external_cell_idxs;
    int my_offset;
} // end class MPISharedField
} // end version(mpi_parallel)

FieldBC create_field_bc(JSONValue field_bc_json, const BoundaryCondition bc, const int[] block_offsets, string conductivity_model_name, int ncells, ExternalCircuit circuit=null){
/*
    Create a field_bc object that will be used later for setting matrix entries near boundaries.
    Currently the data specifying each bc is storied in bc.field_bc as a JSON table.

*/
    string name = getJSONstring(field_bc_json, "name", "not_found");
    FieldBC field_bc;

    switch (name) {
    case "ZeroNormalGradient":
        field_bc = new ZeroNormalGradient();
        break;
    case "FixedField":
        double value = getJSONdouble(field_bc_json, "value", 0.0);
        field_bc = new FixedField(value);
        break;
    case "SheathField":
        double Velectrode = getJSONdouble(field_bc_json, "Velectrode", 0.0);
        string sheath_model = getJSONstring(field_bc_json, "sheath_model", "linear");
        double segment_pitch = getJSONdouble(field_bc_json, "segment_pitch", 0.0);
        double segment_fill = getJSONdouble(field_bc_json, "segment_fill", 1.0);
        double segment_x0 = getJSONdouble(field_bc_json, "segment_x0", 0.0);
        double Ex_applied = getJSONdouble(field_bc_json, "Ex_applied", 0.0);
        double Ex_quad = getJSONdouble(field_bc_json, "Ex_quad", 0.0);
        double Ex_cube = getJSONdouble(field_bc_json, "Ex_cube", 0.0);
        double Ex_ramp_steps = getJSONdouble(field_bc_json, "Ex_ramp_steps", 0.0);
        double Ex_ramp_start = getJSONdouble(field_bc_json, "Ex_ramp_start", 0.0);
        double segment_x1 = getJSONdouble(field_bc_json, "segment_x1", 0.0);
        field_bc = new SheathField(Velectrode, create_sheath_model(sheath_model, field_bc_json),
                                   segment_pitch, segment_fill, segment_x0, Ex_applied, Ex_quad,
                                   Ex_cube, Ex_ramp_steps, Ex_ramp_start, segment_x1);
        break;
    case "CircuitElectrode":
        if (circuit is null)
            throw new Error("A CircuitElectrode boundary condition was requested but no "
                            ~ "config.external_circuit was defined (or it has no nodes).");
        int node = getJSONint(field_bc_json, "node", -1);
        string ce_sheath_model = getJSONstring(field_bc_json, "sheath_model", "linear");
        double ce_pitch = getJSONdouble(field_bc_json, "segment_pitch", 0.0);
        double ce_fill = getJSONdouble(field_bc_json, "segment_fill", 1.0);
        double ce_x0 = getJSONdouble(field_bc_json, "segment_x0", 0.0);
        double ce_x1 = getJSONdouble(field_bc_json, "segment_x1", 0.0);
        field_bc = new CircuitElectrode(circuit, node,
                                        create_sheath_model(ce_sheath_model, field_bc_json),
                                        ce_pitch, ce_fill, ce_x0, ce_x1);
        break;
    case "MixedField":
        double differential = getJSONdouble(field_bc_json, "differential", 1.0);
        double xinsulator = getJSONdouble(field_bc_json, "xinsulator", 0.0);
        double xcollector = getJSONdouble(field_bc_json, "xcollector", 0.0);
        field_bc = new MixedField(differential, xinsulator, xcollector);
        break;
    case "FixedGradient_Test":
        field_bc = new FixedGradient_Test();
        break;
    case "FixedField_Test":
        field_bc = new FixedField_Test();
        break;
    case "unspecified":
        version(mpi_parallel){
            field_bc = new MPISharedField(bc, ncells);
        } else {
            field_bc = new SharedField(bc, block_offsets);
        }
        break;
    default:
        string errMsg = format("The FieldBC '%s' is not available.", name);
        throw new Error(errMsg);
    }
    return field_bc;
}
