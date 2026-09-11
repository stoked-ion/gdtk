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
import std.process : environment;
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
        // Hall discretisationselected via environment so both schemes live in one binary.
        // DEFAULT = central, i.e. the scheme every established result in this project was
        // produced with. The Path 2 upwind split is opt-in via LMR_HALL_SCHEME=upwind
        // until it is validated on a real case; see the Path 2 notes in
        // tools/hall-stencil/. This keeps the committed default bit-identical.
        hall_scheme_central = (environment.get("LMR_HALL_SCHEME", "central") != "upwind");
        // How an insulating (ZeroNormalGradient) boundary is treated once a magnetic
        // field is applied. Only consulted by the upwind (Path 2) scheme; `central`
        // always behaves as "legacy".
        //   tensor : the full J.n = 0 condition for the skew conductivity tensor,
        //            dphi/dn = -beta*dphi/dt + (uxB).n + beta*(uxB).t. Correct for a
        //            genuinely insulating surface sitting in the field.
        //   emf    : beta dropped from the wall condition, dphi/dn = (uxB).n. This is
        //            the condition the module's old TODO named, and it is the right
        //            model for an OPEN end that merely truncates a longer channel: the
        //            Hall current is allowed to continue through the boundary instead of
        //            being turned back into the domain.
        //   legacy : dphi/dn = 0 and no wall source, exactly as before Path 2.
        {
            string m = environment.get("LMR_INSULATOR_BC", "tensor");
            insulator_tensor = (m == "tensor");
            insulator_emf    = (m == "tensor") || (m == "emf");
            if (m != "tensor" && m != "emf" && m != "legacy") {
                throw new Error("LMR_INSULATOR_BC must be tensor, emf or legacy; got '"~m~"'");
            }
        }
        if (GlobalConfig.is_master_task)
            writefln("  [efield/hall] scheme = %s", hall_scheme_central ? "central (original)" : "upwind (Path 2)");
            if (!hall_scheme_central) {
                writefln("  [efield/hall] insulator BC = %s",
                         insulator_tensor ? "tensor" : (insulator_emf ? "emf" : "legacy"));
            }

        // I don't want random bits of the field module hanging off the boundary conditions.
        // Doing it this way is bad encapsulation, but it makes sure that other people only break my code
        // rather than the other way around.
        // External circuit (Path 1). Null unless config.external_circuit declares
        // nodes, in which case every solve takes the augmented-system path below.
        circuit = create_external_circuit(GlobalConfig.external_circuit);

        field_bcs.length = localFluidBlocks.length;
        foreach(i, block; localFluidBlocks){
            field_bcs[i].length = block.bc.length;
            foreach(j, bc; block.bc){
                field_bcs[i][j] = create_field_bc(bc.field_bc, bc, block_offsets, conductivity_model_name, N, circuit);
            }
        }

        // Solvability of the augmented system. Deferred until AFTER the BCs exist,
        // because a node with no resistive path to a supply is still well-posed if it
        // carries electrode faces -- the sheath conductance ties it to the plasma (see
        // ExternalCircuit.isGrounded). That is the physical Hall topology: intermediate
        // segment pairs float, isolated by design from both the supply and each other.
        if (circuit !is null) {
            auto nfaces = new size_t[circuit.nnodes];
            nfaces[] = 0;
            foreach(i, block; localFluidBlocks){
                foreach(j, bc; block.bc){
                    auto celec = cast(CircuitElectrode) field_bcs[i][j];
                    if (celec is null) continue;
                    foreach (f; bc.faces) if (celec.is_electrode(f)) nfaces[celec.nodeId()] += 1;
                }
            }
            version(mpi_parallel){
                // A node's faces can live entirely on another rank, so the count that
                // decides solvability must be the global one.
                auto tmp = new long[nfaces.length];
                foreach (m, v; nfaces) tmp[m] = cast(long) v;
                MPI_Allreduce(MPI_IN_PLACE, tmp.ptr, cast(int) tmp.length, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
                foreach (m, v; tmp) nfaces[m] = cast(size_t) v;
            }
            string why;
            if (!circuit.isGrounded(why, nfaces))
                throw new Error("external_circuit is not solvable: " ~ why);
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

    /*
        Build the vertex-averaged Hall conductivity field used by the conservative Hall
        convection term (Path 2).

        WHY VERTICES. Splitting the tensor sigma_t = sigma_S + sigma_SS (Parent, Shneider
        & Macheret, JCP 230 (2011) 1439-1453, Eqs. 39-42), the skew part contributes a
        face flux S*sigma_H*(t . grad phi) with t = (n_y, -n_x) -- the TANGENTIAL
        derivative of phi along the face. Integrating by parts around the closed cell
        boundary (the [sigma_H*phi] bracket vanishes because the contour is closed):

            contour_int sigma_H (t.grad phi) dS = - contour_int phi (t.grad sigma_H) dS

        and t.grad sigma_H = (-d sigma_H/dy, +d sigma_H/dx) . n = a.n, which is exactly
        Parent's Eq. (41) convection speed. So the Hall term is a CONVECTIVE flux in phi,
        not a diffusive flux in grad phi -- see tools/hall-stencil/fluxform.py, which
        verifies this against the paper's own test cases (the centred form undershoots
        their 10-40 V boundary range by 26 V; this form is exactly monotone).

        The face integral of a.n then telescopes to a difference of sigma_H at the face's
        two ENDPOINTS:

            S*(a.n) = integral_face (t . grad sigma_H) dS = sigma_H(v_end) - sigma_H(v_start)

        with v_start -> v_end running along +t. Two properties follow, and they are the
        whole point of the change:

          1. SINGLE-VALUED. Both cells sharing a face use the same two vertex values and
             opposite n (hence opposite t), so their contributions are exactly equal and
             opposite. The old form needed a tangential GRADIENT reconstructed from each
             cell's own stencil, so the two sides disagreed wherever the stencil family
             changed -- the "discrete curl" defect this module's comments record.
          2. DIVERGENCE-FREE TO MACHINE PRECISION. Summing around a closed cell, each
             vertex appears once as a start and once as an end, so the sum telescopes to
             exactly zero. Analytically div(a) = -d2(sigma_H)/dxdy + d2(sigma_H)/dydx = 0;
             here that identity survives discretisation exactly, so a uniform phi produces
             exactly zero net current regardless of the sigma field.

        Vertex values are the unweighted mean of the cells meeting at the vertex. At a
        BLOCK JOIN the remote cells are included via the halo (below), so both blocks
        average the same set and property 1 still holds across the join -- without that,
        the two sides would compute different vertex values and the flux would be
        two-valued at exactly the 7 joins of a typical X2 channel.
    */
    void computeHallVertexField(FluidBlock[] localFluidBlocks) {
        auto gmodel = GlobalConfig.gmodel_master;
        // The NOMINAL field decides whether the Hall machinery runs at all; the LOCAL
        // field (appliedBzAt) is what enters the physics, so a tapered magnet gives
        // beta -> 0 outside it. With applied_B_ramp = 0 the two are identical everywhere
        // and this is exactly the pre-existing path. appliedBzNominal() (not
        // config.applied_Bz) because a TABULATED profile leaves applied_Bz at zero,
        // which would silently disable Hall.
        double Bz_nom = appliedBzNominal();
        bool hall_on = GlobalConfig.electric_field_hall_effect && (Bz_nom != 0.0);

        if (sigmaH_cell.length != N) sigmaH_cell.length = N;
        sigmaH_cell[] = 0.0;
        if (sigmaH_vtx.length != localFluidBlocks.length) sigmaH_vtx.length = localFluidBlocks.length;

        if (!hall_on) {
            // Leave every vertex value at zero: a.n is then identically zero and the
            // convection term vanishes, recovering the scalar-sigma path exactly.
            foreach(i, block; localFluidBlocks){
                if (sigmaH_vtx[i].length != block.vertices.length) sigmaH_vtx[i].length = block.vertices.length;
                sigmaH_vtx[i][] = 0.0;
            }
            return;
        }

        // 1. sigma_H at cell centres.
        foreach(blkid, block; localFluidBlocks){
            foreach(cell; block.cells){
                int k = cell.id + block_offsets[blkid];
                double sig = conductivity(cell.fs.gas, cell.pos[0], gmodel).re;
                double beta = conductivity.hall_beta(cell.fs.gas, gmodel,
                                                     appliedBzAt(cell.pos[0].x.re));
                sigmaH_cell[k] = sig*beta/(1.0 + beta*beta);
            }
        }

        // 2. Halo: the remote cells just across each shared block boundary. The
        //    exchanger already knows the mapping (it is the same one used for phi in the
        //    matrix-vector product), so we borrow it and copy the result out before the
        //    linear solve overwrites the buffer with phi data.
        version(mpi_parallel){
            exchanger.update_buffers(sigmaH_cell);
            if (sigmaH_halo.length != exchanger.external_cell_buffer.length)
                sigmaH_halo.length = exchanger.external_cell_buffer.length;
            sigmaH_halo[] = exchanger.external_cell_buffer[];
        }

        // 3. Scatter cell values to vertices, then divide by the count.
        foreach(blkid, block; localFluidBlocks){
            if (sigmaH_vtx[blkid].length != block.vertices.length)
                sigmaH_vtx[blkid].length = block.vertices.length;
            sigmaH_vtx[blkid][] = 0.0;
            auto count = new double[block.vertices.length];
            count[] = 0.0;

            foreach(cell; block.cells){
                int k = cell.id + block_offsets[blkid];
                foreach(v; cell.vtx){
                    sigmaH_vtx[blkid][v.id] += sigmaH_cell[k];
                    count[v.id] += 1.0;
                }
            }
            // Remote contributions at shared boundaries, so a join vertex sees the same
            // set of cells from both sides.
            foreach(j, bc; block.bc){
                auto field_bc = field_bcs[blkid][j];
                if (!field_bc.isShared) continue;
                foreach(face; bc.faces){
                    int oid = field_bc.other_id(face);
                    double sH;
                    if (oid < N) {
                        sH = sigmaH_cell[oid];
                    } else {
                        version(mpi_parallel){
                            size_t h = oid - N;
                            if (h >= sigmaH_halo.length) continue;
                            sH = sigmaH_halo[h];
                        } else {
                            continue;
                        }
                    }
                    foreach(v; face.vtx){
                        sigmaH_vtx[blkid][v.id] += sH;
                        count[v.id] += 1.0;
                    }
                }
            }
            foreach(vi; 0 .. block.vertices.length){
                if (count[vi] > 0.0) sigmaH_vtx[blkid][vi] /= count[vi];
            }
        }

        // Raw field values, printed unconditionally on the first solve: if sigma_H is
        // flat the whole Hall term is inert and every downstream number is meaningless.
        if (hall_rowsum_reports == 0) {
            double lo = 1.0e300, hi = -1.0e300;
            size_t nv = 0;
            foreach(blkid, block; localFluidBlocks){
                nv += block.vertices.length;
                foreach(cell; block.cells){
                    double v = sigmaH_cell[cell.id + block_offsets[blkid]];
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }
            }
            if (GlobalConfig.is_master_task) {
                writefln("  [efield/hall] hall_on=%s  sigma_H(cell) in [%.4g, %.4g]  nvtx=%d",
                         hall_on, lo, hi, nv);
                stdout.flush();
            }
        }

        // PURE TELESCOPING CHECK. Independent of the rest of the assembly: for every
        // cell, sum S*(a.n) over ALL its faces. Each vertex is the "end" of one face and
        // the "start" of the next, so the sum must vanish to round-off. If it does not,
        // a uniform potential manufactures current and nothing downstream can be trusted.
        if (hall_rowsum_reports < 3) {
            double worst = 0.0, scale = 0.0;
            foreach(blkid, block; localFluidBlocks){
                foreach(cell; block.cells){
                    double sum = 0.0, mag = 0.0;
                    foreach(io, face; cell.iface){
                        if (face.vtx.length != 2) continue;
                        double nxf = cell.outsign[io]*face.n.x.re;
                        double nyf = cell.outsign[io]*face.n.y.re;
                        double tx = nyf, ty = -nxf;
                        auto v0 = face.vtx[0]; auto v1 = face.vtx[1];
                        double along = (v1.pos[0].x.re - v0.pos[0].x.re)*tx
                                     + (v1.pos[0].y.re - v0.pos[0].y.re)*ty;
                        double sH0 = sigmaH_vtx[blkid][v0.id];
                        double sH1 = sigmaH_vtx[blkid][v1.id];
                        double San = (along >= 0.0) ? (sH1 - sH0) : (sH0 - sH1);
                        sum += San; mag += fabs(San);
                    }
                    if (fabs(sum) > worst) { worst = fabs(sum); scale = mag; }
                }
            }
            version(mpi_parallel){
                double[2] loc = [worst, scale]; double[2] glb;
                MPI_Allreduce(loc.ptr, glb.ptr, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                worst = glb[0]; scale = glb[1];
            }
            // Only report once sigma_H actually varies. On the first solves the flow is
            // still uniform, every vertex difference is zero, and the check is vacuous --
            // it would otherwise burn all three reports before the flow develops.
            if (scale > 0.0) {
                if (GlobalConfig.is_master_task) {
                    writefln("  [efield/hall] telescoping |sum_faces S*(a.n)| = %.3e "
                             ~ "(scale %.3e, relative %.3e)",
                             worst, scale, worst/scale);
                    stdout.flush();
                }
                hall_rowsum_reports += 1;
            }
        }
    }

    void solve_efield(FluidBlock[] localFluidBlocks, bool verbose) {
        A[] = 0.0;
        b[] = 0.0;
        Ai[] = -1;

        // Hall convection field (Path 2). Must run before the assembly, and before the
        // linear solve, which reuses the exchanger's buffer for phi.
        computeHallVertexField(localFluidBlocks);

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
                // Sensitivity of the reconstructed gradient to a prescribed wall slope,
                // C = (_Cx, _Cy); see the insulator boundary condition below. Zero for an
                // interior cell, and C.n == 1 exactly for a wall cell.
                double _Cx = 0.0, _Cy = 0.0;
                int wall_io = -1;

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
                            wall_io = to!int(io);
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

                    _Cx    = mixin(ZGN_Gx)/D;
                    _Cy    = mixin(ZGN_Gy)/D;

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

                    _Cx    = mixin(ZGE_Gx)/D;
                    _Cy    = mixin(ZGE_Gy)/D;

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

                    _Cx    = mixin(ZGS_Gx)/D;
                    _Cy    = mixin(ZGS_Gy)/D;

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

                    _Cx    = mixin(ZGW_Gx)/D;
                    _Cy    = mixin(ZGW_Gy)/D;

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

                double Bz_nom = appliedBzNominal();
                bool hall_on = GlobalConfig.electric_field_hall_effect && (Bz_nom != 0.0);

                // ---------------------------------------------------------------------
                // PATH 2: the insulating-wall boundary condition for the FULL tensor.
                //
                // With a skew sigma_t, J.n = 0 at a bare wall is NOT dphi/dn = 0. Using
                // the identity stated at the face loop below, J.n = m.(-grad phi + uxB)
                // with m = sigma_P*n + sigma_H*t and t = (n_y, -n_x), dividing through by
                // sigma_P gives an oblique-derivative (mixed Robin) condition:
                //
                //     dphi/dn = -beta*dphi/dt + (uxB).n + beta*(uxB).t  ==  g_n
                //
                // beta = 0 recovers dphi/dn = (uxB).n -- the case the module's old TODO
                // named -- and dropping the uxB part too recovers exactly what the ZG*
                // families assume. At beta ~ 15 the tangential term IS the condition:
                // holding dphi/dn = 0 makes the wall a far better insulator than it
                // physically is and piles potential up at the channel ends, which is the
                // +-1-2 kV span C6_pow showed one step after switch-on under the split.
                //
                // Implementation. The ZG* families solve a 4x4 system in which the wall
                // neighbour's row is replaced by the constraint row (n_x, n_y, 0, 0) with
                // a ZERO right-hand side. That system is linear, so a non-zero g_n simply
                // adds g_n*C to the reconstructed gradient, C = (ZG?_Gx, ZG?_Gy)/D, and
                // C.n == 1 identically (C == n on a symmetric stencil). g_n itself
                // depends on dphi/dt, so close the loop once, exactly:
                //
                //     t.grad   = t.grad_h + g_n*gamma,   gamma = C.t
                //     g_n      = (s - beta*t.grad_h)/(1 + beta*gamma)
                //     s        = (uxB).n + beta*(uxB).t
                //
                // and the whole condition collapses into a rescaling of this cell's OWN
                // stencil weights plus one known constant. No new neighbours, no
                // bandwidth change, and no lagging: the tangential coupling is fully
                // implicit. That matters -- at beta = 15 the prescribed derivative
                // direction n + beta*t is within 4 degrees of tangential, so a Picard-
                // lagged version would iterate a fixed point with gain ~beta.
                //
                // Because the SAME corrected weights then feed every use of the
                // reconstruction -- the wall face's own flux and the cell's other faces
                // alike -- there is exactly one place to change and no double counting.
                //
                // UPWIND ONLY. `central` keeps the historical dphi/dn = 0 wall so that
                // every established result stays bit-identical.
                double robin_gx = 0.0, robin_gy = 0.0;
                if (!hall_scheme_central && insulator_emf && wall_io >= 0 && Bz_nom != 0.0) {
                    auto wface = cell.iface[wall_io];
                    double Bz_wall = appliedBzAt(wface.pos.x.re);
                    double wsign = cell.outsign[wall_io];
                    double wnx = wsign*wface.n.x.re;
                    double wny = wsign*wface.n.y.re;
                    double wtx =  wny;
                    double wty = -wnx;
                    double wbeta = (hall_on && insulator_tensor) ?
                        conductivity.hall_beta(wface.fs.gas, gmodel, Bz_wall) : 0.0;
                    // (u x B) = (uy*Bz, -ux*Bz, 0)
                    double wux = wface.fs.vel.x.re;
                    double wuy = wface.fs.vel.y.re;
                    double exn = Bz_wall*(wuy*wnx - wux*wny);
                    double ext = Bz_wall*(wuy*wtx - wux*wty);
                    double s_bc = exn + wbeta*ext;
                    double gamma = _Cx*wtx + _Cy*wty;
                    double den = 1.0 + wbeta*gamma;
                    // gamma vanishes on a symmetric stencil and is small on any smooth
                    // grid, so den ~ 1. Guard anyway rather than let one pathological
                    // cell produce a sign-flipped wall.
                    if (fabs(den) > 0.1) {
                        double f = wbeta/den;
                        double[4] tau;
                        foreach(j; 0 .. 4) tau[j] = wtx*fdx[j] + wty*fdy[j];
                        double tauI = wtx*_Ix + wty*_Iy;
                        foreach(j; 0 .. 4) {
                            fdx[j] -= f*_Cx*tau[j];
                            fdy[j] -= f*_Cy*tau[j];
                        }
                        _Ix -= f*_Cx*tauI;
                        _Iy -= f*_Cy*tauI;
                        // The known part of g_n. fdx/_Ix carry a 1/D that is applied at
                        // the point of use; these do not.
                        robin_gx = _Cx*s_bc/den;
                        robin_gy = _Cy*s_bc/den;
                    }
                }

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
                    // The ZNG conservation gate. It exists because the CENTRAL
                    // tangential-gradient Hall flux is a discrete curl whose defect
                    // concentrates where the stencil family changes, and gating it there
                    // measured ~50x better in level error. It is retained EXACTLY as it was
                    // for the central scheme.
                    //
                    // It is deliberately NOT applied under the Path 2 upwind split, where it
                    // would be actively harmful: that scheme's exactness rests on
                    // sum_faces S*(a.n) = 0 around each cell, which telescopes only if EVERY
                    // face contributes, and gating any face leaves a residue that drives a
                    // spurious current under a uniform phi.
                    if (hall_scheme_central) {
                        if (hall_face && zng_layer[k]) hall_face = false;
                        if (hall_face && !face.is_on_boundary) {
                            auto pcell = (face.left_cell is cell) ? face.right_cell : face.left_cell;
                            if (zng_layer[pcell.id + block_offsets[blkid]]) hall_face = false;
                        }
                    }
                    // The PHYSICAL Hall parameter, never gated. beta sets the Pedersen
                    // conductivity sigma_P = sigma/(1+beta^2), which is a property of the
                    // magnetised plasma and applies on every face including walls; only the
                    // Hall ROTATION (the tangential/convective part) is a discretisation
                    // choice that may be gated.
                    //
                    // The old code folded both into one gated beta, so a gated face silently
                    // reverted to the UNMAGNETISED sigma. With the full tensor that was a
                    // factor |m| = sigma/sqrt(1+beta^2) -> sigma, i.e. ~15x at beta=15. After
                    // the Path 2 split the symmetric part alone is sigma/(1+beta^2), so the
                    // same gate becomes a ~226x conductivity jump at exactly the electrode
                    // walls -- which is what blew C6_pow up one step after switch-on.
                    // CENTRAL keeps the original single gated beta, so its results are
                    // bit-identical to every established run. UPWIND separates the two roles:
                    // beta sets the Pedersen conductivity sigma_P = sigma/(1+beta^2), a
                    // physical property that applies on every face including walls, while
                    // beta_rot (gated) drives only the tensor rotation. Folding both into one
                    // gated beta means a gated face silently reverts to the UNMAGNETISED
                    // sigma -- a factor 1+beta^2 (~226 at beta=15) once the tensor is split.
                    // Local applied field: constant unless a magnet taper is configured.
                    double Bz_app = appliedBzAt(face.pos.x.re);
                    double beta_full = conductivity.hall_beta(face.fs.gas, gmodel, Bz_app);
                    double beta     = hall_scheme_central ? ((hall_face) ? beta_full : 0.0)
                                                          : ((hall_on)   ? beta_full : 0.0);
                    double obb = 1.0/(1.0 + beta*beta);
                    double beta_rot = (hall_face) ? (hall_scheme_central ? beta : beta_full) : 0.0;
                    double obb_rot = 1.0/(1.0 + beta_rot*beta_rot);
                    // FULL tensor rotated normal m = sigma_t^T n. Retained ONLY for the
                    // u x B source below, which is a flux of a known vector field: both
                    // cells sharing a face see the same m, so it is single-valued and
                    // conservative as it stands, and it involves no derivative of phi.
                    // The source tensor is the PHYSICAL one on every face, true domain
                    // boundaries included: this is the flux of a KNOWN field, it is
                    // single-valued, and no conservation argument asks for it to be
                    // gated. The old beta_rot gating silently reverted a boundary face to
                    // the unmagnetised sigma. `central` keeps beta_rot and so stays
                    // bit-identical; under the split the wall needs the true m for the
                    // insulator condition above to balance.
                    double beta_src = hall_scheme_central ? beta_rot : beta;
                    double obb_src  = 1.0/(1.0 + beta_src*beta_src);
                    double mxF = sigmaF*(nxF + beta_src*nyF)*obb_src;
                    double myF = sigmaF*(nyF - beta_src*nxF)*obb_src;
                    // PATH 2 SPLIT. The grad-phi flux keeps only the SYMMETRIC (Pedersen)
                    // part, m_S = sigma_P*n. The skew (Hall) part -- which in the old form
                    // entered here as sigma_H*(t . grad phi), a tangential gradient
                    // reconstructed differently by each of the two cells sharing the face,
                    // and hence the discrete-curl non-conservation -- is moved below to an
                    // exactly conservative upwinded convection term. See
                    // computeHallVertexField for the derivation.
                    // A/B switch for the two Hall discretisations, so both can be run
                    // from one binary and compared directly:
                    //   LMR_HALL_SCHEME=central -> the original full-tensor centred flux
                    //   LMR_HALL_SCHEME=upwind  -> the Path 2 split (default)
                    double sigmaP = sigmaF*obb;
                    double mSx, mSy;
                    if (hall_scheme_central) {
                        mSx = sigmaF*(nxF + beta*nyF)*obb;   // full tensor, as before
                        mSy = sigmaF*(nyF - beta*nxF)*obb;
                    } else {
                        mSx = sigmaP*nxF;                    // symmetric (Pedersen) part only
                        mSy = sigmaP*nyF;
                    }

                    // Hybrid method
                    double facx = nxF - ehatx*ehatx*nxF - ehatx*ehaty*nyF;
                    double facy = nyF - ehaty*ehatx*nxF - ehaty*ehaty*nyF;
                    double fac = (ehatx*nxF + ehaty*nyF)/emag;

                    // sigma-folded (Hall-rotated) stencil factors. These replace the
                    // products sigmaF*facx, sigmaF*facy, sigmaF*fac in the flux terms;
                    // for beta = 0 they are exactly those products. The un-folded
                    // facx/facy/fac remain for the *_direct_component BC calls, which
                    // fold sigma internally.
                    double mdote = ehatx*mSx + ehaty*mSy;
                    double sfacx = mSx - ehatx*mdote;
                    double sfacy = mSy - ehaty*mdote;
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
                            // The face conductivity here must match what the interior
                            // scheme uses, or the insulator boundary conducts at a totally
                            // different rate from the bulk. Note m.n = sigma_P even for the
                            // full tensor, so sigma was always the wrong scale here; the
                            // central scheme merely masks it because its sigma_H tangential
                            // term partly compensates. Under the Path 2 split there is no
                            // such compensation and the mismatch is a factor 1+beta^2 (~226
                            // at beta=15), which drives phi to +-2 kV on a 0-400 V problem.
                            // Only the upwind branch is changed, so `central` stays
                            // bit-identical to the established results.
                            double sig_zng = hall_scheme_central ? sigmaF : sigmaP;
                            sfacx = sig_zng*nxF;
                            sfacy = sig_zng*nyF;
                            sfac = 0.0;
                        } else if (auto celec = cast(CircuitElectrode) field_bc){
                            // Electrode wired into the external circuit. Same
                            // suppression of the gas-conduction stencil as SheathField
                            // (the cold face carries ~no current); the difference is
                            // that the electrode potential is an UNKNOWN, so the
                            // linearization contributes to four blocks rather than two.
                            facx = 0.0; facy = 0.0; fac = 0.0;
                            // PATH 2: zeroing the stencil removes this face's PEDERSEN
                            // flux and its source, which is what "the gas carries no
                            // current here" means for the scalar path. It does NOT remove
                            // the face's HALL flux: under the split that is no longer a
                            // local term at all, it lives in the contour sum -S_an*phi,
                            // and gating any face there breaks the telescoping that makes
                            // a uniform phi drive no current. So subtract the face's own
                            // sigma_H*(t.grad phi) explicitly instead, by giving it the
                            // stencil factor -sigma_H*t. What is left is the sum over the
                            // cell's OTHER faces of the full physical flux, exactly as
                            // wanted; it vanishes for a uniform phi, so the conservation
                            // check still holds; and it is one-sided only at a domain
                            // boundary, where there is no partner cell to be conservative
                            // with. Without this the electrode wall leaks an O(sigma_H)
                            // current and the corner cells where it meets an insulating
                            // end run away -- 4e7 V/m on C6_pow, with the error growing
                            // under refinement (tools/hall-stencil/channel.py, test E).
                            // sigma_H here MUST come from the same vertex field the
                            // convection term uses, not from the face's own gas state.
                            // At a cold electrode the FACE conductivity has collapsed to
                            // ~0 while the vertex average (built from the hot near-wall
                            // CELL values) is the bulk value -- so a face-based sigma_H
                            // subtracts almost nothing and leaves the leak in place.
                            double sigH_w = 0.0;
                            if (!hall_scheme_central && face.vtx.length == 2) {
                                sigH_w = 0.5*(sigmaH_vtx[blkid][face.vtx[0].id]
                                            + sigmaH_vtx[blkid][face.vtx[1].id]);
                            }
                            sfacx = -sigH_w*nyF;
                            sfacy =  sigH_w*nxF;
                            sfac  = 0.0;
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
                            // PATH 2: zeroing the stencil removes this face's PEDERSEN
                            // flux and its source, which is what "the gas carries no
                            // current here" means for the scalar path. It does NOT remove
                            // the face's HALL flux: under the split that is no longer a
                            // local term at all, it lives in the contour sum -S_an*phi,
                            // and gating any face there breaks the telescoping that makes
                            // a uniform phi drive no current. So subtract the face's own
                            // sigma_H*(t.grad phi) explicitly instead, by giving it the
                            // stencil factor -sigma_H*t. What is left is the sum over the
                            // cell's OTHER faces of the full physical flux, exactly as
                            // wanted; it vanishes for a uniform phi, so the conservation
                            // check still holds; and it is one-sided only at a domain
                            // boundary, where there is no partner cell to be conservative
                            // with. Without this the electrode wall leaks an O(sigma_H)
                            // current and the corner cells where it meets an insulating
                            // end run away -- 4e7 V/m on C6_pow, with the error growing
                            // under refinement (tools/hall-stencil/channel.py, test E).
                            // sigma_H here MUST come from the same vertex field the
                            // convection term uses, not from the face's own gas state.
                            // At a cold electrode the FACE conductivity has collapsed to
                            // ~0 while the vertex average (built from the hot near-wall
                            // CELL values) is the bulk value -- so a face-based sigma_H
                            // subtracts almost nothing and leaves the leak in place.
                            double sigH_w = 0.0;
                            if (!hall_scheme_central && face.vtx.length == 2) {
                                sigH_w = 0.5*(sigmaH_vtx[blkid][face.vtx[0].id]
                                            + sigmaH_vtx[blkid][face.vtx[1].id]);
                            }
                            sfacx = -sigH_w*nyF;
                            sfacy =  sigH_w*nxF;
                            sfac  = 0.0;
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

                    // Constant (uxB) part of the insulator Robin slope. Identically zero
                    // unless this cell has a ZNG wall face and the upwind scheme is on.
                    if (robin_gx != 0.0 || robin_gy != 0.0) {
                        b[k] -= S*(sfacx*robin_gx + sfacy*robin_gy);
                    }

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

                    // PATH 2: the Hall term, as an exactly conservative upwinded
                    // convective flux  -S*(a.n)*phi_face  (see computeHallVertexField).
                    //
                    // S*(a.n) is the change in sigma_H between the face's two endpoints,
                    // taken along t = (n_y, -n_x). Because both cells sharing the face
                    // read the same two vertex values and have opposite n, their
                    // contributions cancel exactly; and around a closed cell the sum
                    // telescopes to exactly zero, so a uniform phi drives no current.
                    //
                    // Upwinding: row k accumulates +contour_int (sigma grad phi).n dS, so
                    // its diffusive diagonal is negative. Taking the DONOR cell keeps that
                    // sign, which is what makes the operator monotone:
                    //     a.n > 0 -> phi_face = phi_k      (diagonal gets -S*(a.n) < 0)
                    //     a.n < 0 -> phi_face = phi_other  (off-diagonal gets -S*(a.n) > 0)
                    // This is first-order upwind, i.e. terms 1-3 of Parent's Eq. (45). The
                    // minmod anti-diffusion (term 4) is a nonlinear deferred correction and
                    // is applied on the RHS; see hall_antidiffusion below.
                    // Applied on EVERY face -- interior, block-shared and true domain
                    // boundary alike. That is not optional: the sum of S*(a.n) around a
                    // closed cell telescopes to exactly zero only if no face is skipped,
                    // and that identity is what guarantees a uniform phi drives no current.
                    // An earlier version gated this to interior faces and C6_pow blew up
                    // one step after the field switched on (0.73 -> 2.8e6 -> negative
                    // internal energy), which is exactly the spurious-source signature.
                    if (hall_on && !hall_scheme_central && face.vtx.length == 2) {
                        double tx =  nyF;
                        double ty = -nxF;
                        auto v0 = face.vtx[0];
                        auto v1 = face.vtx[1];
                        double ddx = v1.pos[0].x.re - v0.pos[0].x.re;
                        double ddy = v1.pos[0].y.re - v0.pos[0].y.re;
                        double along = ddx*tx + ddy*ty;   // >0 if v0->v1 runs along +t
                        double sH0 = sigmaH_vtx[blkid][v0.id];
                        double sH1 = sigmaH_vtx[blkid][v1.id];
                        double S_an = (along >= 0.0) ? (sH1 - sH0) : (sH0 - sH1);
                        if (S_an != 0.0) {
                            bool interior = !face.is_on_boundary
                                || field_bcs[blkid][face.bc_id].isShared;
                            if (S_an > 0.0 || !interior) {
                                // Donor is this cell. At a domain boundary we take phi_k
                                // whatever the sign: the exterior carries no independent
                                // potential we can upwind from (a sheath electrode's metal
                                // potential is not the plasma-edge value), and using phi_k
                                // keeps the telescoping exact, since under a uniform phi
                                // every face value is the same constant.
                                A[k*nbands + 2] += -S_an;
                            } else {
                                A[k*nbands + iio] += -S_an;   // donor is the neighbour
                            }
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
                        bool zng_face = false;
                        if (face.is_on_boundary) {
                            auto fbc = field_bcs[blkid][face.bc_id];
                            if ((cast(ZeroNormalGradient) fbc) !is null) {
                                insulator = true;
                                zng_face = true;
                            }
                            if (((cast(SheathField) fbc) !is null) ||
                                ((cast(CircuitElectrode) fbc) !is null)) insulator = true;
                        }
                        // Under the split a ZNG wall DOES carry this source. Its J.n = 0
                        // is imposed through the reconstruction (the Robin slope above),
                        // and for the cell's assembled flux sum to equal the physical one
                        // every face must contribute its m.(uxB) term. Dropping it is what
                        // made the old wall condition inconsistent: it does not converge
                        // under grid refinement at all, see tools/hall-stencil/channel.py.
                        // Sheath and circuit-electrode faces still get no source -- their
                        // gas-conduction stencil is suppressed and their current comes
                        // from the sheath law instead.
                        if (!insulator || (zng_face && !hall_scheme_central && insulator_emf)) {
                            double uxf = face.fs.vel.x.re;
                            double uyf = face.fs.vel.y.re;
                            double mx = mxF, my = myF;
                            if (zng_face) {
                                // Must match the Robin slope imposed above, or the wall's
                                // net flux does not vanish: m_wall = sigma_P*(n + beta*t)
                                // with the SAME beta the condition was written with (zero
                                // in "emf" mode). sigma_P itself always uses the physical
                                // beta -- it is a property of the magnetised gas.
                                double bi = insulator_tensor ? beta : 0.0;
                                mx = sigmaP*(nxF + bi*nyF);
                                my = sigmaP*(nyF - bi*nxF);
                            }
                            // (u x B) = (uy*Bz, -ux*Bz, 0); source = m.(uxB) S
                            b[k] += S * Bz_app * (mx*uyf - my*uxf);
                        }
                    }
                }
            }
        }

        // CONSERVATION CHECK (Path 2). For any cell all of whose faces are interior or
        // block-shared, the assembled row must annihilate a uniform phi: the diffusive
        // terms are differences, the FD-stencil gradient is exact on constants, and the
        // Hall convection sums to -phi_k * sum_faces S*(a.n), which telescopes to zero
        // because each vertex is the "end" of one face and the "start" of the next.
        //
        // This is the single most informative check on the Hall discretisation: a row sum
        // that is not at round-off means a uniform potential is manufacturing current,
        // which is precisely the defect Path 2 exists to remove. Reported for the first
        // few solves only.
        // NB: deliberately NOT gated on `verbose` -- the Newton-Krylov path calls
        // solve_efield with verbose=false (newtonkrylovsolver.d:2539, :3489), which is
        // exactly the path this check matters for. The counter advances identically on
        // every rank, so the reduction below stays collective.
        if (hall_rowsum_reports < 3) {
            double worst = 0.0;
            double scale = 0.0;
            foreach(blkid, block; localFluidBlocks){
                foreach(cell; block.cells){
                    bool all_interior = true;
                    foreach(face; cell.iface){
                        if (face.is_on_boundary && !(field_bcs[blkid][face.bc_id].isShared)) {
                            all_interior = false; break;
                        }
                    }
                    if (!all_interior) continue;
                    int k = cell.id + block_offsets[blkid];
                    double rs = 0.0, mag = 0.0;
                    foreach(band; 0 .. nbands){
                        if (Ai[k*nbands + band] < 0) continue;
                        rs  += A[k*nbands + band];
                        mag += fabs(A[k*nbands + band]);
                    }
                    if (fabs(rs) > worst) { worst = fabs(rs); scale = mag; }
                }
            }
            version(mpi_parallel){
                double[2] loc = [worst, scale]; double[2] glb;
                MPI_Allreduce(loc.ptr, glb.ptr, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                worst = glb[0]; scale = glb[1];
            }
            if (GlobalConfig.is_master_task) {
                writefln("  [efield/hall] interior row-sum |sum A| = %.3e  (row |A| scale %.3e, "
                         ~ "relative %.3e) -- should be at round-off",
                         worst, scale, (scale > 0.0) ? worst/scale : 0.0);
                stdout.flush();
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
            // Report the solved potential range for the first few solves that carry a
            // varying sigma_H. A field solve that has gone wrong shows up here long
            // before the flow crashes, and distinguishes "the field is garbage" from
            // "the field is fine but the J x B source reacts badly to it".
            if (hall_phi_reports < 3) {
                double lo = phi[0], hi = phi[0];
                foreach (v; phi) { if (v < lo) lo = v; if (v > hi) hi = v; }
                version(mpi_parallel){
                    double[2] loc = [-lo, hi]; double[2] glb;
                    MPI_Allreduce(loc.ptr, glb.ptr, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                    lo = -glb[0]; hi = glb[1];
                }
                if (GlobalConfig.is_master_task) {
                    writefln("  [efield/hall] solved phi in [%.4g, %.4g] V", lo, hi);
                    stdout.flush();
                }
                hall_phi_reports += 1;
            }
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
            // Periodic state report. The first-few-solves diagnostic above catches a
            // mis-assembled circuit; this one is for reading the CONVERGED state off the
            // end of a run -- node potentials, supply-leg currents and the total supplied
            // power, which is topology dependent and cannot be reconstructed afterwards
            // from the flow field alone.
            if (circuit_solve_count >= 3 && (circuit_solve_count % 500) == 0
                && GlobalConfig.is_master_task) {
                double[] legI;
                double Psup = circuit.supplyPower(legI);
                writef("  [efield/circuit] state @solve %d:", circuit_solve_count);
                foreach (m; 0 .. K) writef(" q[%d]=%.6g", m, qsol[m]);
                writef("  legI=");
                foreach (i, Il; legI) writef("%s%.6g", i ? "," : "", Il);
                writef("  P_supply=%.6g W/m", Psup);
                double Isrc = circuit.sourceCurrent();
                if (Isrc != 0.0) writef("  I_source=%.6g A/m", Isrc);
                writeln();
                stdout.flush();
            }
            if (circuit_solve_count >= 3) circuit_solve_count++;
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

                // A SheathField/CircuitElectrode face reports phif = 0: the electrode's
                // metal potential is not the plasma-edge value, so the face supplies no
                // usable phi data point. Feeding that 0 into the reconstruction as if it
                // were data puts a spurious phi = 0 Dirichlet half a cell away, and the
                // gradient in the one cell row touching each electrode comes out
                // ~phi_k/(h/2) -- three orders of magnitude too large, and of the wrong
                // sign. The matrix assembly has reconstructed that missing point by linear
                // extrapolation along the line to the opposite neighbour since the Hall
                // terms first activated it; this routine never got the same treatment, so
                // the *reported* E (which the UDF reads to build JxB and the Joule source)
                // stayed wrong in exactly the cells where the current concentrates. Same
                // reconstruction here:
                //     phi_j ~= phi_k + t*(phi_opp - phi_k),  t = (d_j . d_opp)/|d_opp|^2,
                // which is exact on constant and linear fields.
                foreach(io, face; cell.iface){
                    if (!face.is_on_boundary) continue;
                    auto fbc = field_bcs[blkid][face.bc_id];
                    if (((cast(SheathField) fbc) is null)
                        && ((cast(CircuitElectrode) fbc) is null)) continue;
                    size_t iopp = (io+2)%4;
                    auto oface = cell.iface[iopp];
                    bool opp_ok = !oface.is_on_boundary
                        || field_bcs[blkid][oface.bc_id].isShared;
                    double t = 0.0;   // fallback: mirror the cell value (constant-exact)
                    if (opp_ok) {
                        double dopp2 = dx[iopp]*dx[iopp] + dy[iopp]*dy[iopp];
                        t = (dx[io]*dx[iopp] + dy[io]*dy[iopp])/dopp2;
                    }
                    phis[io] = (1.0-t)*cell.electric_potential.re + t*phis[iopp];
                }

                double dxN = dx[0]; double dyN = dy[0]; double nxN = nx[0]; double nyN = ny[0];
                double dxE = dx[1]; double dyE = dy[1]; double nxE = nx[1]; double nyE = ny[1];
                double dxS = dx[2]; double dyS = dy[2]; double nxS = nx[2]; double nyS = ny[2];
                double dxW = dx[3]; double dyW = dy[3]; double nxW = nx[3]; double nyW = ny[3];

                double[4] fdx, fdy;
                double _Ix, _Iy, D;
                // Sensitivity of the reconstructed gradient to a prescribed wall slope,
                // C = (_Cx, _Cy); see the insulator boundary condition below. Zero for an
                // interior cell, and C.n == 1 exactly for a wall cell.
                double _Cx = 0.0, _Cy = 0.0;
                int wall_io = -1;

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
                            wall_io = to!int(io);
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

                    _Cx    = mixin(ZGN_Gx)/D;
                    _Cy    = mixin(ZGN_Gy)/D;

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

                    _Cx    = mixin(ZGE_Gx)/D;
                    _Cy    = mixin(ZGE_Gy)/D;

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

                    _Cx    = mixin(ZGS_Gx)/D;
                    _Cy    = mixin(ZGS_Gy)/D;

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

                    _Cx    = mixin(ZGW_Gx)/D;
                    _Cy    = mixin(ZGW_Gy)/D;

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

                // The ZG* families reconstruct the gradient that satisfies grad(phi).n = 0
                // at the wall. Under the Path 2 split the wall actually satisfies the
                // Robin condition derived in the matrix assembly above, so add the same
                // g_n*C correction here; otherwise the reported field (and the boundary
                // current computed from it) would disagree with the operator that
                // produced phi. `central` is left exactly as it was.
                if (!hall_scheme_central && insulator_emf && wall_io >= 0
                    && appliedBzNominal() != 0.0) {
                    auto wface = cell.iface[wall_io];
                    double Bz_e = appliedBzAt(wface.pos.x.re);
                    double wsign = cell.outsign[wall_io];
                    double wnx = wsign*wface.n.x.re;
                    double wny = wsign*wface.n.y.re;
                    double wtx =  wny;
                    double wty = -wnx;
                    double wbeta = (GlobalConfig.electric_field_hall_effect && insulator_tensor) ?
                        conductivity.hall_beta(wface.fs.gas, GlobalConfig.gmodel_master, Bz_e) : 0.0;
                    double wux = wface.fs.vel.x.re;
                    double wuy = wface.fs.vel.y.re;
                    double s_bc = Bz_e*(wuy*wnx - wux*wny) + wbeta*Bz_e*(wuy*wtx - wux*wty);
                    double den = 1.0 + wbeta*(_Cx*wtx + _Cy*wty);
                    if (fabs(den) > 0.1) {
                        double gn = (s_bc - wbeta*(wtx*Ex + wty*Ey))/den;
                        Ex += _Cx*gn;
                        Ey += _Cy*gn;
                    }
                }

                cell.electric_field[0] = Ex;
                cell.electric_field[1] = Ey;
            }
        }

        // Diagnostic: the largest reconstructed |grad phi| and where it is. A field with
        // an unresolved boundary layer -- the classic magnetised-end-region layer of a
        // Hall device, whose thickness scales like the channel height over beta -- shows
        // up here as a max hugely larger than the bulk V/H scale, long before it shows up
        // as a crash in the flow solver downstream. Reported for the first few solves.
        if (hall_phi_reports <= 4 && appliedBzNominal() != 0.0) {
            double emax = 0.0, ex_at = 0.0, ey_at = 0.0, exv = 0.0, eyv = 0.0;
            foreach(blkid, block; localFluidBlocks){
                foreach(cell; block.cells){
                    double e = sqrt(cell.electric_field[0]^^2 + cell.electric_field[1]^^2);
                    if (e > emax) {
                        emax = e;
                        ex_at = cell.pos[0].x.re; ey_at = cell.pos[0].y.re;
                        exv = cell.electric_field[0]; eyv = cell.electric_field[1];
                    }
                }
            }
            version(mpi_parallel){
                double[5] loc = [emax, ex_at, ey_at, exv, eyv];
                double[5*128] all;
                int nranks; MPI_Comm_size(MPI_COMM_WORLD, &nranks);
                if (nranks <= 128) {
                    MPI_Allgather(loc.ptr, 5, MPI_DOUBLE, all.ptr, 5, MPI_DOUBLE, MPI_COMM_WORLD);
                    foreach(r; 0 .. nranks) {
                        if (all[r*5] > emax) {
                            emax = all[r*5]; ex_at = all[r*5+1]; ey_at = all[r*5+2];
                            exv = all[r*5+3]; eyv = all[r*5+4];
                        }
                    }
                }
            }
            if (GlobalConfig.is_master_task) {
                writefln("  [efield/hall] max |grad phi| = %.4g V/m (%.4g, %.4g) at x=%.5g y=%.5g",
                         emax, exv, eyv, ex_at, ey_at);
                stdout.flush();
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
    int hall_rowsum_reports = 0;
    int hall_phi_reports = 0;
    bool insulator_tensor = true;   // LMR_INSULATOR_BC, see the constructor
    bool insulator_emf    = true;
    bool hall_scheme_central;
    FieldBC[][] field_bcs;
    int[] block_offsets; // FIXME: Badness with the block id's not matching their order in local fluid blocks
    bool[] zng_layer;    // cells with one-sided (non-interior) ZNG stencil families; Hall is gated off on their faces
    // Hall convection field (Path 2). sigma_H = sigma*beta/(1+beta^2) at cells, its
    // halo across block boundaries, and its vertex-averaged values per block. See
    // computeHallVertexField.
    double[] sigmaH_cell;      // [N] global-indexed
    double[] sigmaH_halo;      // [nExtraCells] remote cells, in other_id order
    double[][] sigmaH_vtx;     // [block][vertex id]
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
