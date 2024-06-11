// main.d: a steady laminar flamelet calculator by NNG.

module slf;

import std.stdio;
import std.math;
import std.mathspecial;
import std.format;
import std.algorithm;
import gas;
import gas.physical_constants;
import kinetics;
import nm.smla;
import nm.bbla;
import nm.number;
import nm.complex;

/* Code borrowed from zero_rk:
   Approximate inverse error function implemented from:
   "A handy approximation for the error function and its inverse"
   by Sergei Winitzki
*/

@nogc
double erfc_inv(double q) {
    if(q <= 0.0) return float.infinity;
    if(q >= 2.0) return -float.infinity;
    double p = 1 - q;
    double sgn = p < 0 ? -1 : 1;

    double logpt = log((1 - p)*(1 + p));
    double pt1 = 2/(PI*0.147) + 0.5*logpt;
    double pt2 = 1/(0.147) * logpt;

    return(sgn*sqrt(-pt1 + sqrt(pt1*pt1 - pt2)));
}

@nogc
void second_derivs_from_cent_diffs(double dZ, size_t N, size_t neq, size_t nsp, double[] U0, double[] U1, number[] U, number[] U2nd){
/*
    Use a central difference stencil to get second order derivatives,
    assuming fixed value end conditions

*/
    size_t lft, ctr, rgt;
    ctr = 0*neq;
    rgt = 1*neq;
    foreach(ii; 0 .. neq) U2nd[ctr + ii] = (U[rgt + ii] - 2.0*U[ctr + ii] + U0[ii])/dZ/dZ;

    foreach(i; 1 .. N-1) {
        lft = (i-1)*neq;
        ctr = i*neq;
        rgt = (i+1)*neq;

        foreach(ii; 0 .. neq) U2nd[ctr+ii] = (U[rgt + ii] - 2.0*U[ctr+ii] + U[lft + ii])/dZ/dZ;
    }

    
    lft = (N-2)*neq;
    ctr = (N-1)*neq;
    foreach(ii; 0 .. neq) U2nd[ctr + ii] = (U1[ii] - 2.0*U[ctr + ii] + U[lft + ii])/dZ/dZ;
    return;
}

@nogc
void compute_residual(GasModel gm, ThermochemicalReactor reactor, GasState gs, number[] omegaMi, double p, size_t N, size_t neq, size_t nsp, double[] Z, number[] U, number[] U2nd, number[] R, bool v=false){
/*
    The residual or right hand side is the time derivatives of the equations.
    See Lapointe et al. equations (1) and (2)
*/

    foreach(i; 0 .. N){
        double arg = erfc_inv(2.0*Z[i]);
        double chi = 0.5*exp(-2.0*arg*arg);
        size_t idx = i*neq;
        bool verbose = v &&(i==15);

        gs.T = U[idx+nsp];
        gs.p = p;
        foreach(j, Yj; U[idx .. idx+nsp]) gs.massf[j] = Yj;
        gm.update_thermo_from_pT(gs);
        //if (isNaN(gs.rho) || (gs.rho <= 0.0)) {
        //    throw new Exception(format("Invalid density. Gasstate is %s", gs));
        //}
        reactor.eval_source_terms(gm, gs, omegaMi);
        number cp = gm.Cp(gs);

        R[idx+nsp] = chi/2.0*U2nd[idx+nsp];
        number asdf = 0.0;
        for(int isp=0; isp<nsp; isp++){
            double Mi = gm.mol_masses[isp];
            number hi = gm.enthalpy(gs, isp);

            R[idx+isp] = chi/2.0*U2nd[idx+isp] + omegaMi[isp]/gs.rho;
            debug { if (verbose) writefln("   chi/2.0*U2nd[idx+isp] %e U2nd[idx+isp] %e omegaMi[isp]/gs.rho %e", chi/2.0*U2nd[idx+isp], U2nd[idx+isp], omegaMi[isp]/gs.rho); }
            R[idx+nsp] -= 1.0/gs.rho/cp*omegaMi[isp]*hi;
            asdf -= 1.0/gs.rho/cp*omegaMi[isp]*hi;
        }
        debug{
            if (verbose) writefln("   T= chi/2.0*U2nd[idx+nsp] %e chi %e U2nd %e", chi/2.0*U2nd[idx+nsp], chi, U2nd[idx+nsp]);
            if (verbose) writefln("Computing residual for cell %d Z %e T %e", i, Z[i], gs.T);
            if (verbose) writefln(" Y: %s ", gs.massf);
            if (verbose) writefln("   asdf= %e", asdf);
        }
    }
}

//void compute_jacobian(GasModel gm, ThermochemicalReactor reactor, double[] U0, double[] U1, double dZ, double p, size_t N, size_t neq, size_t nsp, number[] Up, double[] Z, number[] U, number[] U2nd, number[] R, number[] Rp, Matrix!double J){
///*
//    Fill out a sparse matrix with the derivatives of the governing equations
//    computed using real-valued finite differences.
//*/
//    J._data[] = 0.0;
//    foreach(i; 0 .. neq*N) Up[i] = U[i];
//    double eps = 1e-9;
//
//    // we're computing dRi/dUj, with j being the column and i the row index
//    foreach(j; 0 .. neq){
//        // We can perturb every every third cell and compute the residuals in one go.
//        // This is different to how Eilmer does it, where we can compute the residuals
//        // on a subset of cells which are known to be affected by a given perturbation.
//        foreach(loop; 0 .. 3){
//            for (size_t cell=loop; cell<N; cell+=3){
//                size_t idx = cell*neq + j;
//                Up[idx] += eps;
//            }
//
//            second_derivs_from_cent_diffs(dZ, N, neq, nsp, U0, U1, Up, U2nd);
//            compute_residual(gm, reactor, p, N, neq, nsp, Z, Up, U2nd, Rp);
//
//            for (size_t cell=loop; cell<N; cell+=3){
//                size_t lft = (cell-1)*neq;
//                size_t ctr = cell*neq;
//                size_t rgt = (cell+1)*neq;
//                size_t col = ctr + j; 
//
//                if (cell>0){ // Do left cell
//                    foreach(i; 0 .. neq){
//                        double dRdU = (Rp[lft+i] - R[lft+i])/eps;
//                        size_t row = lft + i;
//                        J[row, col] = dRdU;
//                    }
//                }
//
//                // do the centre cell
//                foreach(i; 0 .. neq){
//                    double dRdU = (Rp[ctr+i] - R[ctr+i])/eps;
//                    size_t row = ctr + i;
//                    J[row, col] = dRdU;
//                }
//
//                if (cell<N-1) { // Do right cell
//                    foreach(i; 0 .. neq){
//                        double dRdU = (Rp[rgt+i] - R[rgt+i])/eps;
//                        size_t row = rgt + i;
//                        J[row, col] = dRdU;
//                    }
//                }
//                // finally, undo the perturbation
//                size_t idx = cell*neq + j;
//                Up[idx] -= eps;
//            }
//        }
//    }
//    return;
//}

//void compute_jacobian2(GasModel gm, ThermochemicalReactor reactor, double[] U0, double[] U1, double dZ, double p, size_t N, size_t neq, size_t nsp, number[] Up, double[] Z, number[] U, number[] U2nd, number[] R, number[] Rp, Matrix!double J){
///*
//    Fill out a dense matrix with the derivatives of the governing equations
//    computed using real-valued finite differences.
//*/
//    J._data[] = 0.0;
//    foreach(i; 0 .. neq*N) Up[i] = U[i];
//    double eps = 1e-16;
//
//    // we're computing dRi/dUj, with j being the column and i the row index
//    foreach(j; 0 .. neq*N){
//        Up[j].im = eps;
//
//        second_derivs_from_cent_diffs(dZ, N, neq, nsp, U0, U1, Up, U2nd);
//        compute_residual(gm, reactor, p, N, neq, nsp, Z, Up, U2nd, Rp);
//
//        foreach(i; 0 .. neq*N){
//           double dRdU = (Rp[i].im)/eps;
//           J[i, j] = dRdU;
//        }
//        Up[j].im = 0.0;
//        foreach(i; 0 .. neq*N) Rp[i].im = 0.0;
//    }
//    return;
//}

int main(string[] args)
{
    int exitFlag = 0; // Presume OK in the beginning.
    
    GasModel gm = init_gas_model("gm.lua");
    ThermochemicalReactor reactor = init_thermochemical_reactor(gm, "rr.lua", "");
    GasState gs = GasState(gm);

    uint nsp = gm.n_species;
    uint neq = nsp+1;

    double p = 75e3;
    size_t N = 32;
    size_t n = N*neq;
    double[] Z;
    Z.length = N;
    foreach(i; 1 .. N+1) Z[i-1] = i/(N+1.0);
    double dZ = 1.0/(N+1.0);

    // Boundary Conditions
    double T0 = 300.0; double T1 = 300.0;
    double[] Y0; Y0.length = nsp; foreach(isp; 0 .. nsp) Y0[isp] = 0.0;
    double[] Y1; Y1.length = nsp; foreach(isp; 0 .. nsp) Y1[isp] = 0.0;
    
    Y0[gm.species_index("N2")] = 0.767;
    Y0[gm.species_index("O2")] = 0.233;

    Y1[gm.species_index("N2")] = 0.88;
    Y1[gm.species_index("H2")] = 0.12;

    double[] U0; U0.length = neq; foreach(isp; 0 .. nsp) U0[isp] = Y0[isp]; U0[nsp] = T0;
    double[] U1; U1.length = neq; foreach(isp; 0 .. nsp) U1[isp] = Y1[isp]; U1[nsp] = T1;

    // Initial Guess
    number[] U,Up;
    number[] Ua,Ub,Uc,Ra,Rb,Rc;
    number[] R,Rp;
    number[] U2nd;
    double sigma = 0.1;

    U.length = neq*N;
    Up.length = neq*N;
    U2nd.length = neq*N;
    R.length = (neq)*N;
    Rp.length = (neq)*N;
    Ua.length = (neq)*N;
    Ub.length = (neq)*N;
    Uc.length = (neq)*N;
    Ra.length = (neq)*N;
    Rb.length = (neq)*N;
    Rc.length = (neq)*N;
    number[] omegaMi; omegaMi.length = nsp; // FIXME: GC called here

    // Set up Jacobian. We have a tridiagonal block matrix
    //auto J = new SMatrix!double();
    //J.aa.length = (neq*neq)*3*N - 2*(neq*neq);
    //foreach(i; 0 .. N){
    //    foreach(jj; 0 .. neq){
    //        // The ia array holds where in ja a certain row starts
    //        J.ia ~= J.ja.length; 

    //        // ia holds which column each entry would be in in a real matrix
    //        // we start with the left side block matrix, assuming this isn't
    //        // i==0, which has no neighbour on that side.
    //        if (i>0) {
    //            foreach(ii; 0 .. neq) J.ja ~= (i-1)*neq + ii;
    //        }
    //        // this cell block matrix entries
    //        foreach(ii; 0 .. neq) J.ja ~= i*neq + ii;
    //        // right cell block matrix entries
    //        if (i<N-1) {
    //            foreach(ii; 0 .. neq) J.ja ~= (i+1)*neq + ii;
    //        }
    //        // Then we do all that again for the next row down.
    //    }
    //}
    //writefln("Gotta add one more element to the array: J.ja.length %d J.aa.length %d", J.ja.length, J.aa.length);
    //J.ia ~= J.ja.length;
    //size_t extent = neq*N;
    //auto J = new Matrix!double(extent, extent);

    foreach(i; 0 .. N){
        size_t idx = i*neq;
        double factor = 0.5*tanh(2*6.0*Z[i] - 6.0) + 0.5;
        foreach(isp; 0 .. nsp) U[idx+isp] = (1.0 - factor)*Y0[isp] + factor*Y1[isp];
        U[idx+nsp] = 1500.0*exp(-(Z[i]-0.5)*(Z[i]-0.5)/2.0/sigma/sigma) + 300.0;
    }
    writefln("Initial condition: ");
    foreach(i; 0 .. N){
        size_t idx = neq*i;
        writefln("i =%d T=%e Y=%s", i, U[idx+nsp].re, U[idx .. idx+nsp]);
    }


    second_derivs_from_cent_diffs(dZ, N, neq, nsp, U0, U1, U, U2nd);
    compute_residual(gm, reactor, gs, omegaMi, p, N, neq, nsp, Z, U, U2nd, R, true);

    writefln("Initial residual: ");
    foreach(i; 0 .. N){
        size_t idx = i*neq;
        writefln("i =%d T=%e Y=%s", i, R[idx+nsp].re, R[idx .. idx+nsp]);
    }
    double GR0 = 0.0; foreach(Ri; R) GR0 += Ri.re*Ri.re;
    GR0 = sqrt(GR0);


    // TODO: How do we check this is right?
    writef("T=[");
    foreach(cell; 0 .. N){
        writef("%- 5.1f,", U[cell*neq+nsp].re);
    }
    writefln("]");
    writef("YH2 =[");
    foreach(cell; 0 .. N){
        writef("%- 4.1e,", U[cell*neq+2].re);
    }
    writefln("]");
    writef("YH2O=[");
    foreach(cell; 0 .. N){
        writef("%- 4.1e,", U[cell*neq+7].re);
    }
    writefln("]");
    immutable int maxiters = 400000;
    double dt = 1e-8;
    foreach(iter; 0 .. maxiters) {
        // Let's try computing some derivatives
        //compute_jacobian2(gm, reactor,  U0, U1, dZ, p, N, neq, nsp, Up, Z, U, U2nd, R,  Rp, J);

        // This might not be the best way of doing this but who knows.
        //decompILU0(J);
        //solve(J, R);
        //foreach(i, Ri; R) Rc[i] = Ri.re;
        //auto perm = decomp!double(J);
        //auto x = new Matrix!double(Rc);
        //solve!double(J, x, perm);

        //double global_relaxation_factor = 1.0;
        //double maxdY = -1.0;
        //foreach(cell; 0 .. N){
        //    foreach(j; 0 .. nsp){
        //        double dY = x[neq*cell+j,0].re;
        //        maxdY = fmax(maxdY, fabs(dY));
        //        double relaxation_factor = fmin(1.0, 1e-1/(fabs(dY)+1e-16));
        //        global_relaxation_factor = fmin(global_relaxation_factor, relaxation_factor);
        //    }
        //}
        //writefln("relax factor is %e from dYmax %e", global_relaxation_factor, maxdY);

        //// R is solved in place to be -delta U
        //foreach(i; 0 .. n) U[i] -= global_relaxation_factor*x[i,0];
        //foreach(cell; 0 .. N){
        //    foreach(j; 0 .. nsp){
        //        U[cell*neq + j] = fmin(fmax(U[cell*neq + j], 0.0), 1.0) ;
        //    }
        //}
        bool verbose = false;
        //if (iter%1000==0) verbose=true;

        if (iter==1000){
            GR0 = 0.0; foreach(Ri; R) GR0 += Ri.re*Ri.re;
            GR0 = sqrt(GR0);
        }

        second_derivs_from_cent_diffs(dZ, N, neq, nsp, U0, U1, U, U2nd);
        compute_residual(gm, reactor, gs, omegaMi, p, N, neq, nsp, Z, U, U2nd, R, verbose);

        foreach(i; 0 .. n) Ua[i] = U[i] + dt/2.0*R[i];
        second_derivs_from_cent_diffs(dZ, N, neq, nsp, U0, U1, Ua, U2nd);
        compute_residual(gm, reactor, gs, omegaMi, p, N, neq, nsp, Z, Ua, U2nd, Ra, false);

        foreach(i; 0 .. n) Ub[i] = U[i] + dt/2.0*Ra[i];
        second_derivs_from_cent_diffs(dZ, N, neq, nsp, U0, U1, Ub, U2nd);
        compute_residual(gm, reactor, gs, omegaMi, p, N, neq, nsp, Z, Ub, U2nd, Rb, false);

        foreach(i; 0 .. n) Uc[i] = U[i] + dt*Rb[i];
        second_derivs_from_cent_diffs(dZ, N, neq, nsp, U0, U1, Uc, U2nd);
        compute_residual(gm, reactor, gs, omegaMi, p, N, neq, nsp, Z, Uc, U2nd, Rc, false);

        foreach(i; 0 .. n) U[i] = U[i] + dt/6.0*(R[i] + 2.0*Ra[i] + 2.0*Rb[i] + Rc[i]);

        //foreach(cell; 0 .. N){
        //    foreach(j; 0 .. nsp){
        //        size_t idx = cell*neq;
        //        U[idx+j] = fmin(fmax(U[idx+j], 0.0), 1.0);
        //    }
        //}

        //foreach(cell; 0 .. N){
        //    double Ytotal = 0.0;
        //    foreach(j; 0 .. nsp) Ytotal += U[cell*neq+j];
        //    foreach(j; 0 .. nsp) U[cell*neq+j]/=Ytotal;
        //}

        double GR = 0.0; foreach(Ri; R) GR += Ri.re*Ri.re;
        GR = sqrt(GR);
        double GRR = GR/GR0;
        if (iter%1000==0){ 
            //writef("T=[");
            //foreach(cell; 0 .. N){
            //    writef("%- 5.1f,", U[cell*neq+nsp].re);
            //}
            //writefln("]");
            //writef("YH2 =[");
            //foreach(cell; 0 .. N){
            //    writef("%- 4.1e,", U[cell*neq+2].re);
            //}
            //writefln("]");
            //writef("YH2O=[");
            //foreach(cell; 0 .. N){
            //    writef("%- 4.1e,", U[cell*neq+7].re);
            //}
            //writefln("]");
            //writefln("------");
            //writef("RT=[");
            //foreach(cell; 0 .. N){
            //    writef("%- 5.1f,", R[cell*neq+nsp].re);
            //}
            //writefln("]");
            //writef("RH2 =[");
            //foreach(cell; 0 .. N){
            //    writef("%- 4.1e,", R[cell*neq+2].re);
            //}
            //writefln("]");
            //writef("RH2O=[");
            //foreach(cell; 0 .. N){
            //    writef("%- 4.1e,", R[cell*neq+7].re);
            //}
            //writefln("]");

            writefln("iter %d GR0 %e GR %e GRR %e", iter, GR0, GR, GRR);
        }
        if (GRR<1e-6) break;
        if (iter==maxiters-1) throw new Error(format("Convergence failed after %s iterations, GRR was %e", maxiters, GRR));
    }
    writefln("Done!");


    return exitFlag;
} // end main()