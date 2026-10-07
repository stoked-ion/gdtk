/**
 * Machinery for solving the large sparse matrix associated with a Parabolic Field Problem.
 *
 * Author: Nick Gibbons
 * Version: 2021-05-24: Prototyping
 */

module lmr.efield.efieldgmres;

import std.algorithm;
import std.math;
import std.stdio;
version(mpi_parallel){
    import mpi;
}

import geom;
import nm.smla : SMatrix, decompILU0, iluApply = solve;

import lmr.efield.efieldbc;
import lmr.globalconfig : GlobalConfig;
import lmr.efield.efieldexchange;
import lmr.fvinterface;

class GMResFieldSolver {
    this() {}

    version(mpi_parallel){
        this(Exchanger exchanger) {
            this.exchanger = exchanger;
        }
    }

    // ILU(0) preconditioner (per-rank; block-Jacobi across MPI ranks). The sparsity
    // structure is fixed (grid + 5-band stencil), and ILU(0) introduces no fill-in,
    // so we build the SMatrix ONCE and on later solves just refill its values and
    // re-factor in place. This avoids allocating a fresh SMatrix (+ per-row dup arrays)
    // on every solve, which under the steady NK loop's ~60 solves/step generated GC
    // garbage faster than the conservative collector reclaimed it (RSS creep -> OOM).
    private SMatrix!double Mfact;
    private bool use_ilu = false;
    private bool ilu_built = false;

    void build_ilu_preconditioner(int n, int nb, double[] A, int[] Ai){
        if (!ilu_built) {
            Mfact = new SMatrix!double();
            foreach(i; 0 .. n){
                size_t[19] cols; double[19] vals; int cnt = 0;   // up to 19 bands (3-D with cross-diffusion)
                foreach(j; 0 .. nb){
                    int jj = Ai[i*nb + j];
                    if (jj < 0 || jj >= n) continue; // drop empty + external/MPI columns
                    cols[cnt] = cast(size_t) jj; vals[cnt] = A[i*nb + j]; cnt++;
                }
                // CSR requires ascending column order within a row; insertion sort.
                for (int a=1; a<cnt; a++){
                    size_t cv = cols[a]; double vv = vals[a]; int b = a-1;
                    while (b >= 0 && cols[b] > cv){ cols[b+1]=cols[b]; vals[b+1]=vals[b]; b--; }
                    cols[b+1]=cv; vals[b+1]=vv;
                }
                Mfact.addRow(vals[0..cnt].dup, cols[0..cnt].dup);
            }
            ilu_built = true;
        } else {
            // Refill values into the cached structure (identical column order to the
            // build above, since Ai is constant), overwriting the previous factors.
            size_t pos = 0;
            foreach(i; 0 .. n){
                size_t[19] cols; double[19] vals; int cnt = 0;   // up to 19 bands (3-D with cross-diffusion)
                foreach(j; 0 .. nb){
                    int jj = Ai[i*nb + j];
                    if (jj < 0 || jj >= n) continue;
                    cols[cnt] = cast(size_t) jj; vals[cnt] = A[i*nb + j]; cnt++;
                }
                for (int a=1; a<cnt; a++){
                    size_t cv = cols[a]; double vv = vals[a]; int b = a-1;
                    while (b >= 0 && cols[b] > cv){ cols[b+1]=cols[b]; vals[b+1]=vals[b]; b--; }
                    cols[b+1]=cv; vals[b+1]=vv;
                }
                foreach(k; 0 .. cnt){ Mfact.aa[pos] = vals[k]; pos++; }
            }
        }
        decompILU0!double(Mfact);
        use_ilu = true;
    }

    // Reusable GMRES work buffers (allocated once, sized by nmax_iter x matrix_size,
    // then reused every solve). Members rather than per-call locals to eliminate the
    // ~40 MB/solve allocation churn (q alone is nmax_iter*matrix_size doubles).
    private double[] r, q, y, xold, xnew, xdiff, h, c, s, QT, R, B, Y;
    private bool work_allocated = false;
    private int work_nmax_iter = -1, work_msize = -1;

    void givens_rotation_cs(int i, int j, int n, double[] h, ref double c, ref double s){
        double a = h[j*n + j];
        double b = h[i*n + j];
        double r = sqrt(a*a + b*b);
        c = a/r;
        s = -b/r;
    }
    
    // Note that the caller has responsibility to allocate xd, unlike in the python code.
    void qr_lsq_backward_substitution(int n, int Rdim1, double[] R, double[] b, ref double[] xd){
        xd[] = 0.0;
        for (int k=n-1; k>=0; k--){
            double sum = 0.0;
            for (int i=n-1; i>k; i--){
                sum += R[k*Rdim1+i]*xd[i];
            }
    
            if (fabs(R[k*Rdim1+k])<1e-16){
                xd[k] = 0.0; // Numpy seems to do this to catch singularities. Why though?
            } else {
                xd[k] = (b[k] - sum)/R[k*Rdim1+k];
            }
        }
    }
    
    double dot_product(double[] a, double[] b, int n){
        double sum = 0.0;
        foreach(i; 0 .. n) sum += a[i]*b[i];
        version(mpi_parallel){
            MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        return sum;
    }
    
    double vector_norm(double[] a, int n){
        double sum = 0.0;
        foreach(i; 0 .. n) sum += a[i]*a[i];
        version(mpi_parallel){
            MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        return sqrt(sum);
    }
    
    void banded_matrix_vector_product(int n, int nb, double[] A, int[] Ai, double[] x, ref double[] b){
        b[] = 0.0;

        version(mpi_parallel){ exchanger.update_buffers(x);}

        foreach(i; 0 .. n){
            foreach(j; 0 .. nb){
                int jj = Ai[i*nb+j];
                if (jj<0) continue;

                double xjj;
                version(mpi_parallel){
                    if (jj>=n){
                        xjj = exchanger.external_cell_buffer[jj-n];
                    } else {
                        xjj = x[jj];
                    }
                } else {
                    xjj = x[jj];
                }
                b[i] += A[i*nb+j]*xjj;
            }
        }
    }
    
    void transpose_and_matrix_multiply(int n, int m, double[] A, double[] b, double[] x){
        /*
            Compute x=(A.T).dot(b), where A is a matrix and b is a vector.
    
            Inputs:
             n : Number of rows in matrix A
             m : Number of columns in matrix A
             A : The matrix to be multiplied (n,m)
             b : The vector to be multiplied (n) <- Note the dimension is n!
    
            Outputs:
             x : The result vector (m)
    
            Notes:
             - Like the other routines in this module, we assume the caller is responsible for allocating x 
             - This routine is never called along the distributed axis of the main matrix, so it doesn't
               need to be MPI aware, I think.
        */
    
        // This is the outer loop, iterating over the long axis, the columns of A, typically matrix_size
        for(int i=0; i<m; i++){ 
            // This is the inner loop, iterating over the short axis, the rows of A, typically k+1 in size
            x[i] = 0.0;
            for(int j=0; j<n; j++){
                x[i] += A[j*m + i]*b[j];
            }
        }
    }
    
    void solve(int matrix_size, int nbands, double[] A, int[] Ai, double[] b, double[] x0,
                      double[] xf, int nmax_iter, bool verbose, double tol=1e-12){
        /*
            Generalised Minimal Residual Method for solving linear systems representated by a banded matrix.
    
            Based on: 
             - https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
             - https://en.wikipedia.org/wiki/QR_decomposition
    
            Inputs:
             matrix_size : Number of rows in the matrix being solved for
             nbands      : Number of entries in a band, typically 5 for this application
             A           : Matrix to be solved for  (matrix_size,nbands)
             Ai          : Connectivity matrix specifying where each entry in Ai would be in the full matrix
             b           : RHS matrix (matrix_size)
             x0          : initial guess to begin building Krylov vectors (matrix_size)
             nmax_iter   : Maximum number of iterations to allocate space for
             tol         : Stop when residual falls below this number
    
            Outputs:
             xf          : The final answer vector (matrix_size)
    
            @author: Nick Gibbons
        */
        // LMR_EFIELD_VERBOSE=1 reports every field solve (iterations, true residual).
        static bool vforce = false, vchecked = false;
        if (!vchecked) { import std.process : environment; vforce = (environment.get("LMR_EFIELD_VERBOSE", "0") == "1"); vchecked = true; }
        if (vforce) verbose = GlobalConfig.is_master_task;
        // Restarted GMRES(m) when configured (electric_field_gmres_restart > 0); the
        // original unrestarted solver below otherwise, unchanged.
        if (GlobalConfig.electric_field_gmres_restart > 0) {
            solve_restarted(matrix_size, nbands, A, Ai, b, x0, xf, nmax_iter, verbose);
            return;
        }
        // Allocate the reusable work buffers once (sizes are constant across solves).
        // On subsequent solves these are reused as-is; the per-solve resets below
        // (QT, R, xold, q) re-initialise what the algorithm reads, and h/c/s/y/xnew/
        // xdiff/B/Y are written before they are read each iteration.
        if (!work_allocated || work_nmax_iter != nmax_iter || work_msize != matrix_size) {
            r.length    = matrix_size;
            q.length    = nmax_iter*matrix_size;
            y.length    = matrix_size;
            xold.length = matrix_size;
            xnew.length = matrix_size;
            xdiff.length= matrix_size;
            h.length    = (nmax_iter+1)*nmax_iter;
            c.length    = nmax_iter;
            s.length    = nmax_iter;
            QT.length   = (nmax_iter+1)*(nmax_iter+1);
            R.length    = (nmax_iter+1)*nmax_iter;
            B.length    = nmax_iter+1;
            Y.length    = nmax_iter+1;
            work_allocated = true; work_nmax_iter = nmax_iter; work_msize = matrix_size;
        }
        banded_matrix_vector_product(matrix_size, nbands, A, Ai, x0, r);
        foreach(i; 0 .. matrix_size) r[i] = b[i] - r[i];
        if (use_ilu) iluApply(Mfact, r);   // left preconditioning: r <- M^{-1} r
        double rnorm = vector_norm(r, matrix_size);

        QT[] = 0.0;
        foreach(i; 0 .. nmax_iter+1) QT[i*(nmax_iter+1)+i] = 1.0;
        R[] = 0.0;
        xold[] = x0[];
    
        // First basis vector is created from r
        foreach(i; 0 .. matrix_size) q[i] = r[i]/rnorm;
    
        int niters = nmax_iter;
        int nprint = nmax_iter/40;
        double residual = 1e99;
        bool success = true;
        int k;
    
        for(k=0; k<niters; k++){
            // Perform Arnoldi Iteration to generate basis vectors
            double[] qk = q[k*matrix_size .. (k+1)*matrix_size];
            banded_matrix_vector_product(matrix_size, nbands, A, Ai, qk, y);
            if (use_ilu) iluApply(Mfact, y);   // left preconditioning: y <- M^{-1} (A qk)

            //writeln("qk");
            //writeln(q[k*matrix_size .. (k+1)*matrix_size]);
    
            for(int j; j<k+1; j++){
                double[] qj = q[j*matrix_size .. (j+1)*matrix_size];
                double hjk = dot_product(y, qj, matrix_size);
                h[j*nmax_iter + k] = hjk;
                foreach(p; 0 .. matrix_size) y[p] -= hjk*qj[p];
            }
    
            double hkp1k = vector_norm(y, matrix_size);
            h[(k+1)*nmax_iter + k] = hkp1k;
            if ((hkp1k != 0.0) && (k != nmax_iter-1)){
                foreach(p; 0 .. matrix_size) q[(k+1)*matrix_size + p] = y[p]/hkp1k;
            }
    
            // Now we need to grow the QR matrices that are used for the least squares problem
            int m = k+2;
            int n = k+1;
            // Start by copying the new column of h into R, the upper triangular matrix
            for(int i=0; i<m; i++) R[i*nmax_iter + k] = h[i*nmax_iter + k];
    
            // Now replay the old givens rotations across the new column, excluding the last element
            for(int j=0; j<k; j++){
                int i = j+1;
                double Rjk = R[j*nmax_iter + k];
                double Rik = R[i*nmax_iter + k];
                R[j*nmax_iter + k] = c[j]*Rjk + -s[j]*Rik;
                R[i*nmax_iter + k] = s[j]*Rjk +  c[j]*Rik;
                double Qjn = QT[j*(nmax_iter+1) + n];
                double Qin = QT[i*(nmax_iter+1) + n];
                QT[i*(nmax_iter+1) + n] = s[j]*Qjn + c[j]*Qin;
            }
    
            // Next, compute the new Givens rotation from the last element, and store it for replaying
            //writeln("h");
            //foreach(i; 0 .. k+2) writeln(h[i*nmax_iter .. (i*nmax_iter+k+1)]);
            int j = k;
            int i = k+1;
            givens_rotation_cs(i, j, nmax_iter, R, c[k], s[k]);
            double Rjk = R[j*nmax_iter + k];
            double Rik = R[i*nmax_iter + k];
            R[j*nmax_iter + k] = c[j]*Rjk + -s[j]*Rik;
            R[i*nmax_iter + k] = s[j]*Rjk +  c[j]*Rik;
            for(int p=0; p<m; p++){
                double Qjp = QT[j*(nmax_iter+1) + p];
                double Qip = QT[i*(nmax_iter+1) + p];
                QT[j*(nmax_iter+1) + p] = c[j]*Qjp + -s[j]*Qip;
                QT[i*(nmax_iter+1) + p] = s[j]*Qjp +  c[j]*Qip;
            }
    
            //writeln("QT");
            //foreach(p; 0 .. m) writeln(QT[p*(nmax_iter+1) .. p*(nmax_iter+1)+m]);
            //writeln("R");
            //foreach(p; 0 .. n) writeln(R[p*(nmax_iter) .. p*(nmax_iter)+n]);
    
            // With everything set up, solve the least squared problem and get a new answer vector xnew
            for(int p=0; p<n; p++) B[p] = rnorm*QT[p*(nmax_iter+1) + 0];
            qr_lsq_backward_substitution(n, nmax_iter, R, B, Y);
            transpose_and_matrix_multiply(n, matrix_size, q, Y, xnew);
            for(int p=0; p<matrix_size; p++) xnew[p] += x0[p];

            for(int p=0; p<matrix_size; p++) xdiff[p] = (xnew[p] - xold[p]);
            residual = vector_norm(xdiff, matrix_size);
            xold[] = xnew[];
    
            //if ((k%nprint==0) && verbose) write(".");
            //writefln("iter: %d residual %e hkp1k %e", k, residual, hkp1k);
            // Relative convergence: ||dx|| measured relative to ||x|| (the +1.0 keeps
            // it ~absolute for small-magnitude solutions, e.g. MES verification cases).
            // The original absolute 1e-12 on ||dx|| is unmeetable for large-magnitude
            // fields, such as the ~hundreds-of-volts field driven by the u x B source.
            if (residual < tol*(vector_norm(xnew, matrix_size) + 1.0)) break;
        }
        if (residual >= tol*(vector_norm(xnew, matrix_size) + 1.0)) success=false;
        if (vforce) {
            // the TRUE preconditioned residual of the returned iterate, relative to M^{-1} b
            banded_matrix_vector_product(matrix_size, nbands, A, Ai, xnew, y);
            foreach(i; 0 .. matrix_size) y[i] = b[i] - y[i];
            if (use_ilu) iluApply(Mfact, y);
            double rr_ = vector_norm(y, matrix_size);
            foreach(i; 0 .. matrix_size) y[i] = b[i];
            if (use_ilu) iluApply(Mfact, y);
            double bb_ = vector_norm(y, matrix_size);
            if (verbose) writefln("    [efield/gmres] unrestarted: iters=%d  |dx| criterion met=%s  true rel. residual=%.3e",
                                  k, success, rr_/bb_);
        }
        if (verbose) writefln("    Solve Complete: status=%s  iters=%d/%d  residual=%e/%e", success, k, nmax_iter, residual, tol);
        if (success==false) throw new Error("BGMRes failed to converge!");

        xf[] = xnew[];
        return;
    }

    /*
        Restarted, left-preconditioned GMRES(m): Arnoldi with modified Gram-Schmidt and
        Givens rotations, the Krylov basis rebuilt every m iterations.

        Why. The original solver keeps the WHOLE Krylov history: its memory grows as
        iterations^2 + iterations*N, and it forms the full solution vector at every
        iteration, so its work grows as iterations^2 * N. Fine for a 2-D field of a few
        thousand cells that converges in tens of iterations; not for a 3-D field of 10^5-10^6
        cells. Here memory is (m+1)*N and the solution is formed once per cycle.

        Convergence: the preconditioned residual estimate |g_{j+1}| (exact in exact
        arithmetic) relative to the preconditioned right-hand side, below
        config.electric_field_gmres_rtol. nmax_iter caps the TOTAL iterations.
    */
    private double[] Vr, wr, rr, Hr, cr, sr, gr, yr, hproj;
    private int vr_m = -1, vr_n = -1;

    void solve_restarted(int n, int nb, double[] A, int[] Ai, double[] b, double[] x0,
                         double[] xf, int nmax_iter, bool verbose)
    {
        immutable int m = GlobalConfig.electric_field_gmres_restart;
        immutable double rtol = GlobalConfig.electric_field_gmres_rtol;
        if (vr_m != m || vr_n != n) {
            Vr.length = (m+1)*n; wr.length = n; rr.length = n;
            Hr.length = (m+1)*m; cr.length = m; sr.length = m; gr.length = m+1; yr.length = m; hproj.length = m+1;
            vr_m = m; vr_n = n;
        }
        xf[] = x0[];
        // preconditioned right-hand side norm, the reference for the relative tolerance
        rr[] = b[];
        if (use_ilu) iluApply(Mfact, rr);
        double bnorm = vector_norm(rr, n);
        if (bnorm == 0.0) { xf[] = 0.0; return; }
        int total = 0;
        double resid = 1.0e300;
        bool converged = false;
        while (total < nmax_iter) {
            banded_matrix_vector_product(n, nb, A, Ai, xf, rr);
            foreach (i; 0 .. n) rr[i] = b[i] - rr[i];
            if (use_ilu) iluApply(Mfact, rr);
            double beta = vector_norm(rr, n);
            resid = beta;
            if (beta <= rtol*bnorm) { converged = true; break; }
            foreach (i; 0 .. n) Vr[i] = rr[i]/beta;
            gr[] = 0.0; gr[0] = beta;
            int jj = 0;
            foreach (j; 0 .. m) {
                double[] vj = Vr[j*n .. (j+1)*n];
                banded_matrix_vector_product(n, nb, A, Ai, vj, wr);
                if (use_ilu) iluApply(Mfact, wr);
                // Classical Gram-Schmidt with one reorthogonalisation (CGS2): all j+1
                // projections in ONE global reduction per pass (2 passes), instead of
                // modified Gram-Schmidt's j+1 sequential reductions. Same orthogonality in
                // practice; O(iterations) MPI_Allreduce calls per solve instead of
                // O(iterations^2), which is what limits the field solve at many ranks.
                foreach (i; 0 .. j+1) Hr[i*m + j] = 0.0;
                foreach (pass; 0 .. 2) {
                    foreach (i; 0 .. j+1) {
                        double sum = 0.0;
                        double[] vi = Vr[i*n .. (i+1)*n];
                        foreach (p; 0 .. n) sum += wr[p]*vi[p];
                        hproj[i] = sum;
                    }
                    version(mpi_parallel) {
                        MPI_Allreduce(MPI_IN_PLACE, hproj.ptr, j+1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    }
                    foreach (i; 0 .. j+1) {
                        double hij = hproj[i];
                        Hr[i*m + j] += hij;
                        double[] vi = Vr[i*n .. (i+1)*n];
                        foreach (p; 0 .. n) wr[p] -= hij*vi[p];
                    }
                }
                double hn = vector_norm(wr, n);
                Hr[(j+1)*m + j] = hn;
                if (hn != 0.0) foreach (p; 0 .. n) Vr[(j+1)*n + p] = wr[p]/hn;
                // apply the previous rotations to the new column, then form this one
                foreach (i; 0 .. j) {
                    double t1 = Hr[i*m + j], t2 = Hr[(i+1)*m + j];
                    Hr[i*m + j]     =  cr[i]*t1 + sr[i]*t2;
                    Hr[(i+1)*m + j] = -sr[i]*t1 + cr[i]*t2;
                }
                double a1 = Hr[j*m + j], a2 = Hr[(j+1)*m + j];
                double rho = sqrt(a1*a1 + a2*a2);
                cr[j] = (rho == 0.0) ? 1.0 : a1/rho;
                sr[j] = (rho == 0.0) ? 0.0 : a2/rho;
                Hr[j*m + j] = rho; Hr[(j+1)*m + j] = 0.0;
                gr[j+1] = -sr[j]*gr[j];
                gr[j]   =  cr[j]*gr[j];
                jj = j + 1; total++;
                resid = fabs(gr[j+1]);
                if (resid <= rtol*bnorm || hn == 0.0 || total >= nmax_iter) break;
            }
            // y = H^{-1} g (upper triangular, jj x jj), x += V y
            for (int i = jj-1; i >= 0; i--) {
                double sum = gr[i];
                foreach (q2; i+1 .. jj) sum -= Hr[i*m + q2]*yr[q2];
                yr[i] = (Hr[i*m + i] != 0.0) ? sum/Hr[i*m + i] : 0.0;
            }
            foreach (i; 0 .. jj) {
                double yi = yr[i];
                foreach (p; 0 .. n) xf[p] += yi*Vr[i*n + p];
            }
            if (resid <= rtol*bnorm) { converged = true; break; }
        }
        if (verbose) writefln("    Restarted GMRES(%d): converged=%s  iters=%d/%d  rel. residual=%.3e (target %.1e)",
                              m, converged, total, nmax_iter, resid/bnorm, rtol);
        if (!converged) {
            writefln("    Restarted GMRES(%d) did NOT converge: %d iterations, rel. residual %.3e (target %.1e)",
                     m, total, resid/bnorm, rtol);
            throw new Error("BGMRes failed to converge!");
        }
    }

private:
    version(mpi_parallel){
        Exchanger exchanger;
    }
}

void test_bgmres(){
    int nmax_iter = 5;
    int matrix_size = 5;
    int nbands = 3;

    double[] A = [ 0., -2.,  1.,
                   1.,  4.,  1.,
                   1., -5.,  1.,
                   1.,  5.,  1.,
                   1., -3.,  0.];
          
    int[] Ai = [-1, 0, 1,
                 0, 1, 2,
                 1, 2, 3,
                 2, 3, 4,
                 3, 4,-1];

    auto gmres = new GMResFieldSolver();
    double[] xtarget = [-1.0, 2.0, -1.2, 3.0, 0.8];
    double[] b;
    b.length = 5;
    gmres.banded_matrix_vector_product(5, 3, A, Ai, xtarget, b);
	double[] x;
    double[] x0;
    x.length = 5;
    x0.length = 5;

    x0[] = 0.0;

    gmres.solve(matrix_size, nbands, A, Ai, b, x0, x, nmax_iter, true);
    double error=0.0;
    foreach(p; 0 .. xtarget.length) error = (xtarget[p] - x[p])*(xtarget[p] - x[p]);
    error = sqrt(error);
    writeln("x:");
    writeln(x);
    writeln("xtarget:");
    writeln(xtarget);
    writeln("error: ", error);
}
