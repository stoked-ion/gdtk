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
         double segment_pitch=0.0, double segment_fill=1.0, double segment_x0=0.0) {
        this.Velectrode = Velectrode;
        this.model = model;
        this.segment_pitch = segment_pitch;
        this.segment_fill = segment_fill;
        this.segment_x0 = segment_x0;
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
    @nogc final bool is_electrode(const FVInterface face) const {
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
        double dV = cell.electric_potential.re - Velectrode;
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
        if (phi_cell != phi_cell) phi_cell = Velectrode;
        double dV0 = phi_cell - Velectrode;
        double J0 = model.current(dV0, face.fs.gas, gm);
        double Jp = model.conductance(dV0, face.fs.gas, gm);
        a_diag = -S*Jp;
        b_rhs  = S*(J0 - Jp*phi_cell);
    }
    override string toString() const {
        return format("SheathField(Velectrode=%g, segment_pitch=%g, segment_fill=%g)",
                      Velectrode, segment_pitch, segment_fill);
    }
private:
    double Velectrode;
    SheathModel model;
    double segment_pitch, segment_fill, segment_x0;
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

FieldBC create_field_bc(JSONValue field_bc_json, const BoundaryCondition bc, const int[] block_offsets, string conductivity_model_name, int ncells){
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
        field_bc = new SheathField(Velectrode, create_sheath_model(sheath_model, field_bc_json),
                                   segment_pitch, segment_fill, segment_x0);
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
