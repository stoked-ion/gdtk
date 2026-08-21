/**
 * Machinery for solving electromagnetic fields through the fluid domain
 *
 * Author: Nick Gibbons
 * Version: 2021-01-20: Prototyping
 */

module lmr.efield.efield;

import std.algorithm;
import std.conv;
import std.format;
import std.math;
import std.stdio;
version(mpi_parallel){
    import mpi;
}

import geom;
import nm.number;
import ntypes.complex;

import lmr.efield.efieldbc;
import lmr.efield.efieldcircuit;
import lmr.efield.efieldconductivity;
import lmr.efield.efieldderivatives;
import lmr.efield.efieldexchange;
import lmr.efield.efieldgmres;
import lmr.fluidblock;
import lmr.fluidfvcell;
import lmr.fvinterface;
import lmr.globalconfig;

immutable uint ZNG_interior = 0b0000;
immutable uint ZNG_north    = 0b0001;
immutable uint ZNG_east     = 0b0010;
immutable uint ZNG_south    = 0b0100;
immutable uint ZNG_west     = 0b1000;
immutable uint[4] ZNG_types = [ZNG_north, ZNG_east, ZNG_south, ZNG_west];

class ElectricField {
    this(const FluidBlock[] localFluidBlocks, const string conductivity_model_name) {
        N = 0;
        foreach(block; localFluidBlocks){
            block_offsets ~= N;
            N += to!int(block.cells.length);
        }
        A.length = N*nbands;
        Ai.length = N*nbands;
        b.length = N;

        if (GlobalConfig.electric_field_gmres_iters > 0) {
            max_iter = GlobalConfig.electric_field_gmres_iters;
        } else {
            max_iter = N;
            version(mpi_parallel) {
                MPI_Allreduce(MPI_IN_PLACE, &max_iter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            }
        }

        phi.length = N;
        phi0.length = N;
        auto gmodel = GlobalConfig.gmodel_master;
        conductivity = create_conductivity_model(conductivity_model_name, gmodel);

        // I don't want random bits of the field module hanging off the boundary conditions.
        // Doing it this way is bad encapsulation, but it makes sure that other people only break my code
        // rather than the other way around.
        // External circuit (Path 1). Null unless config.external_circuit declares
        // nodes, in which case every solve takes the augmented-system path below.
        circuit = create_external_circuit(GlobalConfig.external_circuit);
        if (circuit !is null) {
            string why;
            if (!circuit.isGrounded(why))
                throw new Error("external_circuit is not solvable: " ~ why);
        }

        field_bcs.length = localFluidBlocks.length;
        foreach(i, block; localFluidBlocks){
            field_bcs[i].length = block.bc.length;
            foreach(j, bc; block.bc){
                field_bcs[i][j] = create_field_bc(bc.field_bc, bc, block_offsets, conductivity_model_name, N, circuit);
            }
        }

        version(mpi_parallel){
            exchanger = new Exchanger(GlobalConfig.mpi_rank_for_local_task, GlobalConfig.mpi_size, MPISharedField.instances);
            gmres = new GMResFieldSolver(exchanger);
        } else {
            gmres = new GMResFieldSolver();
        }

        // Cells whose FD stencils use the one-sided ZNG derivative families
        // (celltype != ZNG_interior). The Hall tensor terms are gated off on every
        // face touching one of these cells: their tangential-gradient estimates
        // differ from the interior R_* family at leading order, so a face shared
        // between the two families assembles non-cancelling fluxes (see the
        // conservation gate in solve_efield).
        zng_layer.length = N;
        zng_layer[] = false;
        foreach(i, block; localFluidBlocks){
            foreach(cell; block.cells){
                foreach(face; cell.iface){
                    if (face.is_on_boundary &&
                        ((cast(ZeroNormalGradient) field_bcs[i][face.bc_id]) !is null)) {
                        zng_layer[cell.id + block_offsets[i]] = true;
                    }
                }
            }
        }

        return;
    }

    void solve_efield(FluidBlock[] localFluidBlocks, bool verbose) {
        A[] = 0.0;
        b[] = 0.0;
        Ai[] = -1;

        // Border blocks of the augmented system (Path 1). Only touched when a
        // circuit is present; otherwise everything below is exactly as before.
        immutable size_t K = (circuit is null) ? 0 : circuit.nnodes;
        if (circuit !is null) {
            if (Uc.length != K) {
                Uc.length = K; Wt.length = K;
                foreach (m; 0 .. K) { Uc[m].length = N; Wt[m].length = N; }
                Lmat.length = K*K; cvec.length = K;
            }
            foreach (m; 0 .. K) { Uc[m][] = 0.0; Wt[m][] = 0.0; }
            // start from the network's own stamp; the sheath faces add to it below
            circuit.assemble_L_and_c(Lmat, cvec);
        }

        FluidFVCell other;
        foreach(blkid, block; localFluidBlocks){
            auto gmodel = block.myConfig.gmodel;
            foreach(cell; block.cells){

                // Set up the sparse matrix indexes. This actually only needs to be done once, technically
                int k = cell.id + block_offsets[blkid];
                Ai[nbands*k + 2] = k;

                // We need the distances to the unknown phi locations, which may be faces or cells
                double[4] dx, dy, nx, ny;
                foreach(io, face; cell.iface){
                    double sign = cell.outsign[io];
                    Vector3 pos;
                    int iio = (io>1) ? to!int(io+1) : to!int(io); // -> [0,1,3,4] since 2 is the entry for "cell"

                    if (face.is_on_boundary) {
                        auto field_bc = field_bcs[blkid][face.bc_id];
                        pos = field_bc.other_pos(face);
                        Ai[k*nbands + iio] = field_bc.other_id(face);
                    } else {
                        other = face.left_cell;
                        if (other==cell) other = face.right_cell;
                        pos = other.pos[0];
                        Ai[k*nbands + iio] = other.id + block_offsets[blkid];
                    }

                    nx[io] = sign*face.n.x.re;
                    ny[io] = sign*face.n.y.re;
                    dx[io] = pos.x.re - cell.pos[0].x.re;
                    dy[io] = pos.y.re - cell.pos[0].y.re;
                }

                double dxN = dx[0]; double dyN = dy[0];
                double dxE = dx[1]; double dyE = dy[1];
                double dxS = dx[2]; double dyS = dy[2];
                double dxW = dx[3]; double dyW = dy[3];

                double nxN = nx[0]; double nyN = ny[0];
                double nxE = nx[1]; double nyE = ny[1];
                double nxS = nx[2]; double nyS = ny[2];
                double nxW = nx[3]; double nyW = ny[3];

                double[4] fdx, fdy, fdxx, fdyy;
                double _Ix, _Iy, _Ixx, _Iyy, D;

                // Figure out what kind of cell we are, since ones with ZeroNormalGradient
                // have different equations for the derivatives...
				uint celltype = ZNG_interior;
                foreach(io, face; cell.iface){
					if (face.is_on_boundary) {
						auto field_bc = field_bcs[blkid][face.bc_id];
						// NOTE: SheathField is deliberately NOT folded into the ZNG celltype
						// here. Doing so makes electrode/inflow corner cells have two ZNG-type
						// faces, an unhandled combined celltype. The sheath face's gas-gradient
						// contribution is already suppressed in the matrix assembly (facx=facy
						// =fac=0), and the electrode cell is UDF-guarded, so the R-formula
						// reconstruction there is harmless.
						if ((cast(ZeroNormalGradient) field_bc) !is null) {
                            celltype = celltype | ZNG_types[io];
						}
                    }
				}

                switch (celltype) {
                case ZNG_north:
                    D = mixin(ZGN_D);
                    fdx[0] = mixin(ZGN_Nx);
                    fdx[1] = mixin(ZGN_Ex);
                    fdx[2] = mixin(ZGN_Sx);
                    fdx[3] = mixin(ZGN_Wx);
                    _Ix    = mixin(ZGN_Ix);

                    fdy[0] = mixin(ZGN_Ny);
                    fdy[1] = mixin(ZGN_Ey);
                    fdy[2] = mixin(ZGN_Sy);
                    fdy[3] = mixin(ZGN_Wy);
                    _Iy    = mixin(ZGN_Iy);

                    break;
                case ZNG_east:
                    D = mixin(ZGE_D);
                    fdx[0] = mixin(ZGE_Nx);
                    fdx[1] = mixin(ZGE_Ex);
                    fdx[2] = mixin(ZGE_Sx);
                    fdx[3] = mixin(ZGE_Wx);
                    _Ix    = mixin(ZGE_Ix);

                    fdy[0] = mixin(ZGE_Ny);
                    fdy[1] = mixin(ZGE_Ey);
                    fdy[2] = mixin(ZGE_Sy);
                    fdy[3] = mixin(ZGE_Wy);
                    _Iy    = mixin(ZGE_Iy);

                    break;
                case ZNG_south:
                    D = mixin(ZGS_D);
                    fdx[0] = mixin(ZGS_Nx);
                    fdx[1] = mixin(ZGS_Ex);
                    fdx[2] = mixin(ZGS_Sx);
                    fdx[3] = mixin(ZGS_Wx);
                    _Ix    = mixin(ZGS_Ix);

                    fdy[0] = mixin(ZGS_Ny);
                    fdy[1] = mixin(ZGS_Ey);
                    fdy[2] = mixin(ZGS_Sy);
                    fdy[3] = mixin(ZGS_Wy);
                    _Iy    = mixin(ZGS_Iy);

                    break;
                case ZNG_west:
                    D = mixin(ZGW_D);
                    fdx[0] = mixin(ZGW_Nx);
                    fdx[1] = mixin(ZGW_Ex);
                    fdx[2] = mixin(ZGW_Sx);
                    fdx[3] = mixin(ZGW_Wx);
                    _Ix    = mixin(ZGW_Ix);

                    fdy[0] = mixin(ZGW_Ny);
                    fdy[1] = mixin(ZGW_Ey);
                    fdy[2] = mixin(ZGW_Sy);
                    fdy[3] = mixin(ZGW_Wy);
                    _Iy    = mixin(ZGW_Iy);

                    break;
                case ZNG_interior:
                    D = mixin(R_D);
                    fdx[0] = mixin(R_Nx);
                    fdx[1] = mixin(R_Ex);
                    fdx[2] = mixin(R_Sx);
                    fdx[3] = mixin(R_Wx);
                    _Ix    = mixin(R_Ix);

                    fdy[0] = mixin(R_Ny);
                    fdy[1] = mixin(R_Ey);
                    fdy[2] = mixin(R_Sy);
                    fdy[3] = mixin(R_Wy);
                    _Iy    = mixin(R_Iy);

                    break;
                default:
                    string errMsg = format("An invalid ZNGtype '%s' was requested.", celltype);
                    throw new Error(errMsg);
                }

                double Bz_app = GlobalConfig.applied_Bz;
                bool hall_on = GlobalConfig.electric_field_hall_effect && (Bz_app != 0.0);

                foreach(io, face; cell.iface){
                    int iio = (io>1) ? to!int(io+1) : to!int(io); // -> [0,1,3,4] since 2 is
                    face.fs.gas.sigma = conductivity(face.fs.gas, face.pos, gmodel); // TODO: Redundant work.
                    double sign = cell.outsign[io];
                    double S = face.length.re;
                    double sigmaF = face.fs.gas.sigma.re;
                    double dxF = face.pos.x.re - cell.pos[0].x.re;
                    double dyF = face.pos.y.re - cell.pos[0].y.re;
                    double nxF = sign*face.n.x.re;
                    double nyF = sign*face.n.y.re;
                    double emag = sqrt(dx[io]*dx[io] + dy[io]*dy[io]);
                    double ehatx = dx[io]/emag;
                    double ehaty = dy[io]/emag;

                    // Tensor (Hall) conductivity. With B = Bz z and Hall parameter
                    // beta = e*Bz/(m_e*nu_e) (from the conductivity model), Ohm's law
                    // J = sigma_t (-grad phi + uxB) has the 2x2 tensor
                    //     sigma_t = sigma/(1+beta^2) [[1, -beta], [beta, 1]],
                    // and the face current is J.n = m.(-grad phi + uxB) with the
                    // sigma-weighted rotated normal m = sigma_t^T n:
                    //     mx = sigma_P*nx + sigma_H*ny,  my = sigma_P*ny - sigma_H*nx,
                    // sigma_P = sigma/(1+beta^2), sigma_H = sigma*beta/(1+beta^2).
                    // beta = 0 recovers m = sigma*n (the scalar path) exactly. The
                    // tangential-gradient information m needs is already carried by the
                    // hybrid stencil, so the 5-band matrix structure is unchanged.
                    // Block-to-block (shared) faces are physically interior and get the
                    // rotation; true domain boundaries keep the scalar path -- their
                    // current is set by the BC (sheath law / Dirichlet / ZNG insulator).
                    bool hall_face = hall_on;
                    if (face.is_on_boundary && !(field_bcs[blkid][face.bc_id].isShared)) hall_face = false;
                    // Conservation gate: the tangential Hall flux is a discrete-curl term
                    // whose column sums cancel only around closed, consistent stencil
                    // loops; wherever the loop breaks (a stencil-family change at the
                    // one-sided ZNG cells, the sheath-wall redirect, or any beta jump)
                    // an O(sigma_H) net-current defect is left behind. With the break at
                    // the ZNG corner cells -- where the ZGW/ZGE one-sided FD family and
                    // the wall redirect compound -- the manufactured current reached
                    // ~400 A/m on the X2-ABLE channel, pushing the floating level off by
                    // +1.5 kV. Gating the Hall rotation (beta = 0, scalar sigma) on every
                    // face touching a ZNG-layer cell relocates the break to plain interior
                    // cells, which measures ~50x smaller in level error (-31 V first
                    // solve, -2 V once the sheath Robin anchoring is iterated). The gate
                    // must be symmetric -- both rows of a face must see the same beta --
                    // hence the partner-cell lookup. Remote partners across block
                    // boundaries are assumed interior: ZNG layers normally sit at domain
                    // ends, not at block joins.
                    if (hall_face && zng_layer[k]) hall_face = false;
                    if (hall_face && !face.is_on_boundary) {
                        auto pcell = (face.left_cell is cell) ? face.right_cell : face.left_cell;
                        if (zng_layer[pcell.id + block_offsets[blkid]]) hall_face = false;
                    }
                    double beta = (hall_face) ? conductivity.hall_beta(face.fs.gas, gmodel, Bz_app) : 0.0;
                    double obb = 1.0/(1.0 + beta*beta);
                    double mxF = sigmaF*(nxF + beta*nyF)*obb;
                    double myF = sigmaF*(nyF - beta*nxF)*obb;

                    // Hybrid method
                    double facx = nxF - ehatx*ehatx*nxF - ehatx*ehaty*nyF;
                    double facy = nyF - ehaty*ehatx*nxF - ehaty*ehaty*nyF;
                    double fac = (ehatx*nxF + ehaty*nyF)/emag;

                    // sigma-folded (Hall-rotated) stencil factors. These replace the
                    // products sigmaF*facx, sigmaF*facy, sigmaF*fac in the flux terms;
                    // for beta = 0 they are exactly those products. The un-folded
                    // facx/facy/fac remain for the *_direct_component BC calls, which
                    // fold sigma internally.
                    double mdote = ehatx*mxF + ehaty*myF;
                    double sfacx = mxF - ehatx*mdote;
                    double sfacy = myF - ehaty*mdote;
                    double sfac  = mdote/emag;

                    // Finite difference stencil
                    //    double facx = nxF;
                    //    double facy = nyF;
                    //    double fac = 0.0;
                    //// Direct normal gradient only
                    //    double facx = 0.0;
                    //    double facy = 0.0;
                    //    double fac = (ehatx*nxF + ehaty*nyF)/emag;

                    // PART ONE: Using the hybrid stencil, we first have a direct contribution to the
                    // flux, based on the cell and the other cell on the other side of the face
                    if (face.is_on_boundary) {
                        auto field_bc = field_bcs[blkid][face.bc_id];

                        // For a ZNG boundary, we want to use the stencil gradients for face i's
                        // contribution to the fluxes. It's ugly, but works for the moment
                        if ((cast(ZeroNormalGradient) field_bc) !is null){
                            facx = nxF;
                            facy = nyF;
                            fac = 0.0;
                            sfacx = sigmaF*nxF;
                            sfacy = sigmaF*nyF;
                            sfac = 0.0;
                        } else if (auto celec = cast(CircuitElectrode) field_bc){
                            // Electrode wired into the external circuit. Same
                            // suppression of the gas-conduction stencil as SheathField
                            // (the cold face carries ~no current); the difference is
                            // that the electrode potential is an UNKNOWN, so the
                            // linearization contributes to four blocks rather than two.
                            facx = 0.0; facy = 0.0; fac = 0.0;
                            sfacx = 0.0; sfacy = 0.0; sfac = 0.0;
                            if (celec.is_electrode(face)) {
                                double S_f = face.length.re;
                                double q_star = celec.Velectrode_at(face);
                                double phi_star = cell.electric_potential.re;
                                if (phi_star != phi_star) phi_star = q_star; // NaN guard
                                double J0, Jp;
                                celec.sheathCurrentAndConductance(face, phi_star, gmodel, J0, Jp);
                                double ad, uc, bc_, wt, ld, cn;
                                sheathFaceStamp(S_f, J0, Jp, phi_star, q_star,
                                                ad, uc, bc_, wt, ld, cn);
                                int m = celec.nodeId();
                                A[k*nbands + 2] += ad;
                                b[k]            += bc_;
                                Uc[m][k]        += uc;
                                Wt[m][k]        += wt;
                                Lmat[m*K + m]   += ld;
                                cvec[m]         += cn;
                            }
                        } else if (auto sheath = cast(SheathField) field_bc){
                            // Electrode sheath: the gas carries ~no current at the cold
                            // electrode face (face sigma ~ 0), so suppress its gas-conduction
                            // stencil (facx=facy=fac=0). The electrode current is the sheath
                            // Robin term from the pluggable SheathModel, linearized about the
                            // current plasma-edge potential (NOT scaled by gas sigma).
                            // Segmented electrodes: insulator strips between segments get
                            // no Robin term either, leaving J.n = 0 there.
                            facx = 0.0;
                            facy = 0.0;
                            fac = 0.0;
                            sfacx = 0.0;
                            sfacy = 0.0;
                            sfac = 0.0;
                            if (sheath.is_electrode(face)) {
                                double a_diag, b_rhs;
                                sheath.linearized_robin(face, cell.electric_potential.re, gmodel, a_diag, b_rhs);
                                A[k*nbands + 2] += a_diag;
                                b[k]            += b_rhs;
                            }
                        } else if (field_bc.isShared) {
                            // Block-to-block face: physically interior, so apply the same
                            // (Hall-rotated) direct term as the interior branch. For beta=0
                            // this equals the SharedField lhs_direct/other components.
                            A[k*nbands + 2]  += -1.0*S*sfac;
                            A[k*nbands + iio]+=  1.0*S*sfac;
                        } else {
                            A[k*nbands + 2]  += field_bc.lhs_direct_component(fac, face);
                            A[k*nbands + iio]+= field_bc.lhs_other_component(fac, face);
                            b[k]             -= field_bc.rhs_direct_component(sign, fac, face);
                        }
                    } else {
                        A[k*nbands + 2] +=  -1.0*S*sfac;
                        A[k*nbands + iio]+=  1.0*S*sfac;
                    }

                    // PART TWO: The other part of the gradient comes from a finite difference stencil,
                    // which has components from all of the nearby cells, and cell k:
                    A[k*nbands + 2] +=  S/D*(sfacx*(_Ix) + sfacy*(_Iy));

                    // Each jface makes a contribution to the flux through "face"
                    foreach(jo, jface; cell.iface){
                        int jjo = (jo>1) ? to!int(jo+1) : to!int(jo); // -> [0,1,3,4] since 2 is the entry for "cell"
                        if (jface.is_on_boundary) {
                            auto field_bc = field_bcs[blkid][jface.bc_id];
                            if (((cast(SheathField) field_bc) !is null) ||
                                ((cast(CircuitElectrode) field_bc) !is null)) {
                                // The sheath wall supplies no phi data point (its stencil
                                // components are zero) and the cell keeps the interior R_*
                                // weights (see the celltype note above), so dropping this
                                // term acts like a spurious phi=0 Dirichlet in the cell's
                                // FD gradient: the estimate picks up ~phi_k/h of gauge-
                                // dependent garbage. Dormant while sfacx=sfacy~0 (scalar
                                // sigma, orthogonal grid), it is activated by the Hall
                                // terms and wrecks the field near the electrodes.
                                // Reconstruct the missing point by linear extrapolation
                                // along the line to the opposite neighbour:
                                //   phi_j ~= phi_k + t*(phi_opp - phi_k),
                                //   t = (d_j . d_opp)/|d_opp|^2   (t < 0),
                                // which restores exactness on constant AND linear fields.
                                double w = S/D*(sfacx*fdx[jo] + sfacy*fdy[jo]);
                                if (w != 0.0) {
                                    size_t jopp = (jo+2)%4;
                                    auto oface = cell.iface[jopp];
                                    bool opp_ok = !oface.is_on_boundary
                                        || field_bcs[blkid][oface.bc_id].isShared;
                                    double t = 0.0; // fallback: mirror (constant-exact only)
                                    if (opp_ok) {
                                        double dopp2 = dx[jopp]*dx[jopp] + dy[jopp]*dy[jopp];
                                        t = (dx[jo]*dx[jopp] + dy[jo]*dy[jopp])/dopp2;
                                    }
                                    int jjopp = (jopp>1) ? to!int(jopp+1) : to!int(jopp);
                                    A[k*nbands + 2]     += (1.0-t)*w;
                                    if (t != 0.0) A[k*nbands + jjopp] += t*w;
                                }
                            } else {
                                // The *_stencil_component implementations are linear in
                                // (facx, facy) and do not fold sigma internally, so passing
                                // the sigma-folded factors replaces the external sigmaF.
                                A[k*nbands + jjo] += S*field_bc.lhs_stencil_component(D, sfacx, sfacy, fdx[jo], fdy[jo], jface);
                                b[k]              -= S*field_bc.rhs_stencil_component(D, sfacx, sfacy, fdx[jo], fdy[jo], jface);
                            }
                        } else {
                            A[k*nbands + jjo] += S/D*(sfacx*(fdx[jo])
                                                    + sfacy*(fdy[jo]));
                        }
                    }

                    // u x B motional-EMF source (low magnetic Reynolds number):
                    // charge continuity div(sigma_t(grad phi - uxB)) = 0 puts the
                    // m.(uxB) flux on the RHS (m = sigma_t^T n; scalar sigma*n when the
                    // Hall effect is off). Insulator (ZeroNormalGradient) faces carry no
                    // current, so they get no source. B is the uniform applied field (z).
                    // [TODO: an insulator BC should strictly enforce grad(phi).n =
                    // (uxB).n; this is exact only where (uxB).n ~ 0, as at the
                    // inflow/outflow boundaries of an axial-flow channel.]
                    if (Bz_app != 0.0) {
                        bool insulator = false;
                        if (face.is_on_boundary) {
                            auto fbc = field_bcs[blkid][face.bc_id];
                            if (((cast(ZeroNormalGradient) fbc) !is null) ||
                                ((cast(SheathField) fbc) !is null) ||
                                ((cast(CircuitElectrode) fbc) !is null)) insulator = true;
                        }
                        if (!insulator) {
                            double uxf = face.fs.vel.x.re;
                            double uyf = face.fs.vel.y.re;
                            // (u x B) = (uy*Bz, -ux*Bz, 0); source = m.(uxB) S
                            b[k] += S * Bz_app * (mxF*uyf - myF*uxf);
                        }
                    }
                }
            }
        }

        // The Jacobi Preconditioner, a very simple scheme. Good for diagonally dominant matrices
        if (precondition){
            foreach(blkid, block; localFluidBlocks){
                foreach(cell; block.cells){
                    int k = cell.id + block_offsets[blkid];
                    //writefln(" A[i,:]=[%e,%e,%e,%e,%e] b=[%e]", A[k*nbands+0], A[k*nbands+1], A[k*nbands+2], A[k*nbands+3], A[k*nbands+4], b[k]);
                    double Akk = A[k*nbands + 2];
                    b[k] /= Akk;
                    foreach(iio; 0 .. nbands) A[k*nbands + iio] /= Akk;
                    // Uc is part of THIS SAME ROW of the augmented system, so it must
                    // take the identical row scaling. (Leaving it unscaled makes the
                    // circuit coupling wrong by a factor of Akk per row -- the field
                    // then sees an electrode potential scaled by the local diagonal,
                    // which diverges within a few steps.) Wt/Lmat/cvec belong to the
                    // circuit ROWS, not the cell rows, so they are untouched here.
                    if (circuit !is null) foreach (m; 0 .. K) Uc[m][k] /= Akk;
                }
            }
        }

        // ILU(0) preconditioner (block-Jacobi across MPI ranks), built from the
        // Jacobi-scaled banded matrix so the diagonal is ~1 for stable pivots.
        // Far stronger than point-Jacobi alone for the variable-conductivity Poisson.
        gmres.build_ilu_preconditioner(N, nbands, A, Ai);

        if (circuit is null) {
            // ---- ordinary path: byte-for-byte what this solver has always done ----
            phi0[] = 0.0;
            gmres.solve(N, nbands, A, Ai, b, phi0, phi, max_iter, verbose);
        } else {
            // ---- augmented (external-circuit) path: Woodbury/Schur ----
            // The border blocks are rank-local: a node's faces may live on cells owned
            // by different ranks, so Uc/Wt/Lmat/cvec must be summed across ranks before
            // the Schur complement is formed. (Skipping this is silently correct on one
            // rank and silently WRONG on many -- each rank would solve for a different
            // q and the reconstructed phi would be globally inconsistent.)
            version(mpi_parallel) {
                // Uc and Wt are indexed by LOCAL cell id and must NOT be reduced
                // element-wise -- that would add together unrelated cells on different
                // ranks. Only quantities that are genuinely global get reduced: the
                // K x K / K dot products (handled inside schurSolve via the delegate
                // below) and the sheath contributions to L and c, whose faces may live
                // on any rank.
                //
                // Lmat/cvec include the network stamp, which every rank computed
                // identically; reduce only the sheath contributions by subtracting the
                // common part, summing, then adding it back once.
                double[] Lnet, cnet;
                circuit.assemble_L_and_c(Lnet, cnet);
                foreach (i; 0 .. K*K) Lmat[i] -= Lnet[i];
                foreach (i; 0 .. K)   cvec[i] -= cnet[i];
                MPI_Allreduce(MPI_IN_PLACE, Lmat.ptr, to!int(K*K), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(MPI_IN_PLACE, cvec.ptr, to!int(K),   MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                foreach (i; 0 .. K*K) Lmat[i] += Lnet[i];
                foreach (i; 0 .. K)   cvec[i] += cnet[i];
            }

            // K+1 solves of the UNMODIFIED banded system. The preconditioner is built
            // once above and reused for all of them.
            auto solveA0 = delegate double[](const(double)[] rhs) {
                auto rr = new double[N];
                foreach (i; 0 .. N) rr[i] = rhs[i];
                auto xx = new double[N];
                auto x0 = new double[N]; x0[] = 0.0;
                gmres.solve(N, nbands, A, Ai, rr, x0, xx, max_iter, false);
                return xx;
            };
            void delegate(double[]) reducer = null;
            version(mpi_parallel) {
                reducer = delegate void(double[] buf) {
                    MPI_Allreduce(MPI_IN_PLACE, buf.ptr, to!int(buf.length),
                                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                };
            }
            double[] qsol;
            double[] phisol;
            schurSolve(solveA0, Uc, Wt, Lmat, b, cvec, phisol, qsol, reducer);
            foreach (i; 0 .. N) phi[i] = phisol[i];
            // Every rank solved the same tiny K x K system redundantly, so all ranks
            // now hold identical q. Store it for the next solve's Velectrode_at().
            circuit.setQ(qsol);
            // Diagnostic on the first few solves: a circuit that is mis-assembled
            // usually shows up here as q far from the nominal node voltages, long
            // before the flow solver reports trouble.
            if (circuit_solve_count < 3) {
                writef("  [efield/circuit] solve %d:", circuit_solve_count);
                foreach (m; 0 .. K) writef(" q[%d]=%.6g", m, qsol[m]);
                double pmin = phi[0], pmax = phi[0];
                foreach (i; 0 .. N) { if (phi[i] < pmin) pmin = phi[i]; if (phi[i] > pmax) pmax = phi[i]; }
                writefln("   phi in [%.6g, %.6g]", pmin, pmax);
                circuit_solve_count++;
            }
            if (verbose) {
                writef("    circuit node potentials:");
                foreach (m; 0 .. K) writef(" q[%d]=%.6g", m, qsol[m]);
                writeln();
            }
        }

        // Unpack the solution into the "electric_potential" members stored in the cells
        size_t i = 0;
        foreach(block; localFluidBlocks){
            foreach(cell; block.cells){
                cell.electric_potential = phi[i];
                i += 1;
            }
        }

        // We also have enough info to set the ghost cell potentials too
        version(mpi_parallel){ exchanger.update_buffers(phi);}

        foreach(blkid, block; localFluidBlocks){
            foreach(j, bc; block.bc){
                auto field_bc = field_bcs[blkid][j];
                if (!field_bc.isShared) continue;

                foreach(fidx, f; bc.faces){
                    // Idx is the real cell's location in the phi array
                    int idx = field_bc.other_id(f);

                    double phiidx;
                    version(mpi_parallel) {
                        phiidx = exchanger.external_cell_buffer[idx-phi.length];
                    } else {
                        phiidx = phi[idx];
                    }

                    // Figure out which side the ghost cell is on and set 
                    if (bc.outsigns[fidx] == 1) {
                        f.right_cell.electric_potential = phiidx;
                    } else {
                        f.left_cell.electric_potential = phiidx;
                    }
                }
            }
        }
        return;
    }

    void compute_electric_field_vector(FluidBlock[] localFluidBlocks) {
    /*
        With the electric potential field solved for, compute its gradients
        and store the electric field vector.

        Notes:
         - TODO: MPI

        @author: Nick Gibbons
    */

        FluidFVCell other;
        foreach(blkid, block; localFluidBlocks){
            foreach(cell; block.cells){
                double[4] dx, dy, nx, ny, phis;

                foreach(io, face; cell.iface){
                    Vector3 pos;
                    double phi;
                    double sign = cell.outsign[io];
                    if (face.is_on_boundary) {
                        auto field_bc = field_bcs[blkid][face.bc_id];
                        phi = field_bc.phif(face);
                        pos = field_bc.other_pos(face);
                    } else {
                        other = face.left_cell;
                        if (other==cell) other = face.right_cell;
                        pos = other.pos[0];
                        phi = other.electric_potential;
                    }
                    nx[io] = sign*face.n.x.re;
                    ny[io] = sign*face.n.y.re;
                    dx[io] = pos.x.re - cell.pos[0].x.re;
                    dy[io] = pos.y.re - cell.pos[0].y.re;
                    phis[io] = phi;
                }

                double dxN = dx[0]; double dyN = dy[0]; double nxN = nx[0]; double nyN = ny[0];
                double dxE = dx[1]; double dyE = dy[1]; double nxE = nx[1]; double nyE = ny[1];
                double dxS = dx[2]; double dyS = dy[2]; double nxS = nx[2]; double nyS = ny[2];
                double dxW = dx[3]; double dyW = dy[3]; double nxW = nx[3]; double nyW = ny[3];

                double[4] fdx, fdy;
                double _Ix, _Iy, D;

                // Figure out what kind of cell we are, since ones with ZeroNormalGradient
                // have different equations for the derivatives...
				uint celltype = ZNG_interior;
                foreach(io, face; cell.iface){
					if (face.is_on_boundary) {
						auto field_bc = field_bcs[blkid][face.bc_id];
						// NOTE: SheathField is deliberately NOT folded into the ZNG celltype
						// here. Doing so makes electrode/inflow corner cells have two ZNG-type
						// faces, an unhandled combined celltype. The sheath face's gas-gradient
						// contribution is already suppressed in the matrix assembly (facx=facy
						// =fac=0), and the electrode cell is UDF-guarded, so the R-formula
						// reconstruction there is harmless.
						if ((cast(ZeroNormalGradient) field_bc) !is null) {
                            celltype = celltype | ZNG_types[io];
						}
                    }
				}

                switch (celltype) {
                case ZNG_north:
                    D = mixin(ZGN_D);
                    fdx[0] = mixin(ZGN_Nx);
                    fdx[1] = mixin(ZGN_Ex);
                    fdx[2] = mixin(ZGN_Sx);
                    fdx[3] = mixin(ZGN_Wx);
                    _Ix    = mixin(ZGN_Ix);

                    fdy[0] = mixin(ZGN_Ny);
                    fdy[1] = mixin(ZGN_Ey);
                    fdy[2] = mixin(ZGN_Sy);
                    fdy[3] = mixin(ZGN_Wy);
                    _Iy    = mixin(ZGN_Iy);

                    break;
                case ZNG_east:
                    D = mixin(ZGE_D);
                    fdx[0] = mixin(ZGE_Nx);
                    fdx[1] = mixin(ZGE_Ex);
                    fdx[2] = mixin(ZGE_Sx);
                    fdx[3] = mixin(ZGE_Wx);
                    _Ix    = mixin(ZGE_Ix);

                    fdy[0] = mixin(ZGE_Ny);
                    fdy[1] = mixin(ZGE_Ey);
                    fdy[2] = mixin(ZGE_Sy);
                    fdy[3] = mixin(ZGE_Wy);
                    _Iy    = mixin(ZGE_Iy);

                    break;
                case ZNG_south:
                    D = mixin(ZGS_D);
                    fdx[0] = mixin(ZGS_Nx);
                    fdx[1] = mixin(ZGS_Ex);
                    fdx[2] = mixin(ZGS_Sx);
                    fdx[3] = mixin(ZGS_Wx);
                    _Ix    = mixin(ZGS_Ix);

                    fdy[0] = mixin(ZGS_Ny);
                    fdy[1] = mixin(ZGS_Ey);
                    fdy[2] = mixin(ZGS_Sy);
                    fdy[3] = mixin(ZGS_Wy);
                    _Iy    = mixin(ZGS_Iy);

                    break;
                case ZNG_west:
                    D = mixin(ZGW_D);
                    fdx[0] = mixin(ZGW_Nx);
                    fdx[1] = mixin(ZGW_Ex);
                    fdx[2] = mixin(ZGW_Sx);
                    fdx[3] = mixin(ZGW_Wx);
                    _Ix    = mixin(ZGW_Ix);

                    fdy[0] = mixin(ZGW_Ny);
                    fdy[1] = mixin(ZGW_Ey);
                    fdy[2] = mixin(ZGW_Sy);
                    fdy[3] = mixin(ZGW_Wy);
                    _Iy    = mixin(ZGW_Iy);

                    break;
                case ZNG_interior:
                    D = mixin(R_D);
                    fdx[0] = mixin(R_Nx);
                    fdx[1] = mixin(R_Ex);
                    fdx[2] = mixin(R_Sx);
                    fdx[3] = mixin(R_Wx);
                    _Ix    = mixin(R_Ix);

                    fdy[0] = mixin(R_Ny);
                    fdy[1] = mixin(R_Ey);
                    fdy[2] = mixin(R_Sy);
                    fdy[3] = mixin(R_Wy);
                    _Iy    = mixin(R_Iy);

                    break;
                default:
                    string errMsg = format("An invalid ZNGtype '%s' was requested.", celltype);
                    throw new Error(errMsg);
                }

                double Ex = (_Ix*cell.electric_potential + fdx[0]*phis[0] + fdx[1]*phis[1] + fdx[2]*phis[2] + fdx[3]*phis[3])/D;
                double Ey = (_Iy*cell.electric_potential + fdy[0]*phis[0] + fdy[1]*phis[1] + fdy[2]*phis[2] + fdy[3]*phis[3])/D;
                cell.electric_field[0] = Ex;
                cell.electric_field[1] = Ey;
            }
        }
    }

    void compute_boundary_current(FluidBlock[] localFluidBlocks, ref double current_in, ref double current_out) {
    /*
        Loop over the boundaries of the domain and compute the total electrical current flowing in and out.
        We put the contributions of each face into different buckets depending on their sign, negative means
        current flow in and positive out.

        Notes:
         - TODO: MPI
         - Caution: We assume that the field and conductivity are already set

        @author: Nick Gibbons
    */

        double Iin = 0.0;
        double Iout = 0.0;

        foreach(blkid, block; localFluidBlocks){
            foreach(cell; block.cells){
                foreach(io, face; cell.iface){
                    if (!face.is_on_boundary) continue;

                    auto field_bc = field_bcs[blkid][face.bc_id];
                    double phif = field_bc.phif(face);
                    if (phif==0.0) continue; // FIXME: We don't really need this
                    if (field_bc.isShared) continue;

                    double sign = cell.outsign[io];
                    double S = face.length.re;
                    double sigmaF = face.fs.gas.sigma.re;
                    double dxF = face.pos.x.re - cell.pos[0].x.re;
                    double dyF = face.pos.y.re - cell.pos[0].y.re;
                    double nxF = sign*face.n.x.re;
                    double nyF = sign*face.n.y.re;
                    double emag = sqrt(dxF*dxF + dyF*dyF);
                    double ehatx = dxF/emag;
                    double ehaty = dyF/emag;

                    // Hybrid method
                    double facx = nxF - ehatx*ehatx*nxF - ehatx*ehaty*nyF;
                    double facy = nyF - ehaty*ehatx*nxF - ehaty*ehaty*nyF;
                    double fac = (ehatx*nxF + ehaty*nyF)/emag;

                    double phigrad_dot_n = nxF*cell.electric_field[0] + nyF*cell.electric_field[1];

                    //double phigrad_dot_n = facx*cell.electric_field[0] + 
                    //                       facy*cell.electric_field[1] + 
                    //                       fac*(phif - cell.electric_potential);

                    double I = phigrad_dot_n*S*sigmaF;
                    if (I<0.0) {
                        Iin -= I;
                    } else if (I>0.0) {
                        Iout += I;
                    }
                }
            }
        }
        version(mpi_parallel){
            MPI_Allreduce(MPI_IN_PLACE, &Iin, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &Iout, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        current_in = Iin;
        current_out = Iout;
	}

    void compute_boundary_current_old(FluidBlock[] localFluidBlocks) {
    /*
        Loop over the boundaries of the domain and compute the total electrical current flowing in and out.
        We put the contributions of each face into different buckets depending on their sign, negative means
        current flow in and positive out.

        Notes:
         - TODO: MPI

        @author: Nick Gibbons
    */
        double Iin = 0.0;
        double Iout = 0.0;

        writeln("Called field.compute_boundary_current() ...");
        foreach(blkid, block; localFluidBlocks){
            auto gmodel = block.myConfig.gmodel;
            foreach(cell; block.cells){
                foreach(io, face; cell.iface){
                    if (face.is_on_boundary) {
                        face.fs.gas.sigma = conductivity(face.fs.gas, face.pos, gmodel);
                        double sign = cell.outsign[io];
                        auto field_bc = field_bcs[blkid][face.bc_id];
                        double I = field_bc.compute_current(sign, face, cell);
                        if (I<0.0) {
                            Iin -= I;
                        } else if (I>0.0) {
                            Iout += I;
                        }
                    }
                }
            }
        }
        writefln("    Current in:  %f (A/m)", Iin);
        writefln("    Current out: %f (A/m)", Iout);
	}
private:
    immutable int nbands = 5; // 5 for a 2D structured grid
    immutable bool precondition = true;


    GMResFieldSolver gmres;
    ConductivityModel conductivity;
    ExternalCircuit circuit;      // null => no circuit; the ordinary code path
    double[][] Uc, Wt;            // [K][N] border blocks of the augmented system
    double[] Lmat, cvec;          // K*K row-major, and length K
    int circuit_solve_count = 0;
    FieldBC[][] field_bcs;
    int[] block_offsets; // FIXME: Badness with the block id's not matching their order in local fluid blocks
    bool[] zng_layer;    // cells with one-sided (non-interior) ZNG stencil families; Hall is gated off on their faces
    version(mpi_parallel){
        Exchanger exchanger;
    }

    double[] A;
    int[] Ai;
    double[] b;
    double[] phi,phi0;
    int max_iter;
    int N;
}
