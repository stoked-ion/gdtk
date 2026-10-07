/**
 * efieldstencil.d
 *
 * Dimension-generic gradient reconstruction for the electric-field solver.
 *
 * The 2D solver reconstructs a cell's potential gradient from its four face neighbours with
 * closed forms machine-generated from Maxima (efieldderivatives.d). Those closed forms are
 * the solution of a small linear system, recovered and checked in tools/hall-stencil:
 *
 *   for each face j, neighbour offset d_j = x_j - x_cell (a cell centre or a face centre):
 *       data row:   phi_j - phi_cell = g.d_j + sum_a h_a d_ja^2      (g = grad phi, h_a = d2phi/dx_a^2 terms)
 *   for a face on an insulating (ZeroNormalGradient) wall:
 *       slope row:  g.n_j = g_n                                      (g_n = 0 for the legacy wall)
 *
 * with 2*nd unknowns (g, h) and 2*nd faces, so the system is square: 4 x 4 in 2D (the R_* and
 * ZG* families) and 6 x 6 in 3D. Solving it numerically per cell, once, replaces every
 * hand-derived family at once -- including the edge and corner cells of a 3D grid, which
 * touch two or three walls and would otherwise each need a family of their own (2D already
 * throws on a cell with two insulating faces).
 *
 * The result is a set of weights W[a][j]:  g_a = sum_j W[a][j] * r_j,  where r_j is
 * (phi_j - phi_cell) for a data row and g_n for a slope row. For a data row this is exactly
 * the closed-form weight <fam>_<j><a>/<fam>_D; summing the data weights gives minus the
 * closed-form centre weight <fam>_I<a>/<fam>_D; and the slope-row column is the wall-slope
 * sensitivity C = (<fam>_Gx, <fam>_Gy)/<fam>_D. The unit tests below check all three against
 * efieldderivatives.d on random, deliberately skewed stencils.
 *
 * Author: 2026-10, gdtk-mhd branch.
 */

module lmr.efield.efieldstencil;

import std.math;

/// Maximum faces per cell (hexahedron).
enum int MAXF = 6;

/**
    Solve the reconstruction system for one cell.

    nd       : spatial dimensions, 2 or 3; the cell has nf = 2*nd faces.
    d[j]     : offset from the cell centre to face j's data point (only the first nd used).
    n[j]     : outward unit normal of face j (used only for slope rows).
    slope[j] : true if face j is an insulating wall (slope row), false for a data row.
    W        : output, W[a][j] for a < nd, j < nf.

    Returns false if the system is singular (the caller must then fall back or fail).

    Numerics: the rows are scaled by a characteristic length h (the mean offset), so the
    data rows become (d/h, (d/h)^2) and the slope rows (n, 0); unknowns are then (h g, h^2 h_a),
    all O(1). Gaussian elimination with partial pivoting on the scaled matrix.
*/
@nogc nothrow
bool reconstructionWeights(int nd, const ref double[3][MAXF] d, const ref double[3][MAXF] n,
                           const ref bool[MAXF] slope, ref double[MAXF][3] W)
{
    immutable int nf = 2*nd;
    double h = 0.0; int nh = 0;
    foreach (j; 0 .. nf) {
        if (slope[j]) continue;
        double s = 0.0;
        foreach (a; 0 .. nd) s += d[j][a]*d[j][a];
        h += sqrt(s); nh++;
    }
    if (nh == 0 || !(h > 0.0)) return false;
    h /= nh;

    // M (nf x nf) augmented with the identity, Gauss-Jordan -> M^{-1}.
    double[2*MAXF][MAXF] M;
    foreach (j; 0 .. nf) {
        foreach (c; 0 .. 2*nf) M[j][c] = 0.0;
        if (slope[j]) {
            foreach (a; 0 .. nd) M[j][a] = n[j][a];
        } else {
            foreach (a; 0 .. nd) {
                double t = d[j][a]/h;
                M[j][a] = t;
                M[j][nd + a] = t*t;
            }
        }
        M[j][nf + j] = 1.0;
    }
    foreach (col; 0 .. nf) {
        int piv = col; double big = fabs(M[col][col]);
        foreach (r; col+1 .. nf) if (fabs(M[r][col]) > big) { big = fabs(M[r][col]); piv = r; }
        if (!(big > 1.0e-12)) return false;
        if (piv != col) foreach (c; 0 .. 2*nf) { double t = M[col][c]; M[col][c] = M[piv][c]; M[piv][c] = t; }
        double p = M[col][col];
        foreach (c; 0 .. 2*nf) M[col][c] /= p;
        foreach (r; 0 .. nf) {
            if (r == col) continue;
            double f = M[r][col];
            if (f == 0.0) continue;
            foreach (c; 0 .. 2*nf) M[r][c] -= f*M[col][c];
        }
    }
    // Unknown a of the scaled system is h*g_a, so g_a = (1/h) * sum_j Minv[a][j] * r'_j, with
    // r'_j = r_j for a data row and r'_j = h*g_n for a slope row (its row was not divided by h).
    foreach (a; 0 .. nd) {
        foreach (j; 0 .. nf) {
            double minv = M[a][nf + j];
            W[a][j] = slope[j] ? minv : minv/h;
        }
    }
    return true;
}

/// lmr's structured face order (geom Face enum): west, east, south, north, bottom, top,
/// i.e. i-, i+, j-, j+, k-, k+. Face j lies on axis j/2; its opposite is j xor 1.
@nogc nothrow pure int faceAxisOf(int j) { return j/2; }
@nogc nothrow pure int oppositeFace(int j) { return j ^ 1; }


/// Band layout of a cell's row in the banded matrix: the diagonal sits in the middle, the
/// faces fill the slots around it. nf = 4 gives the 2D solver's [0,1,3,4] with diagonal 2.
@nogc nothrow pure int diagBand(int nf) { return nf/2; }
@nogc nothrow pure int faceBand(int j, int nf) { return (j < nf/2) ? j : j + 1; }

version(efieldstencil_test) {
    import std.stdio;
    import std.random;
    import lmr.efield.efieldderivatives;

    // Evaluate one 2D family against the generic solve on a random skewed stencil.
    // wall = -1 for the interior R_* family, else the index (N=0,E=1,S=2,W=3) of the
    // insulating face (ZG? family). Returns the worst relative mismatch over every weight.
    double check2d(ref Random rng, int wall)
    {
        double[2][4] base = [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]];
        double[3][MAXF] dd; double[3][MAXF] nn; bool[MAXF] sl;
        foreach (j; 0 .. 4) {
            dd[j][0] = 1.0e-3*(base[j][0] + uniform(-0.3, 0.3, rng));
            dd[j][1] = 1.0e-3*(base[j][1] + uniform(-0.3, 0.3, rng));
            dd[j][2] = 0.0;
            sl[j] = (j == wall);
        }
        double th = uniform(0.0, 2.0*PI, rng);
        foreach (j; 0 .. 4) { nn[j][0] = cos(th); nn[j][1] = sin(th); nn[j][2] = 0.0; }
        double[MAXF][3] Wg;
        assert(reconstructionWeights(2, dd, nn, sl, Wg));

        double dxN = dd[0][0], dyN = dd[0][1], dxE = dd[1][0], dyE = dd[1][1];
        double dxS = dd[2][0], dyS = dd[2][1], dxW = dd[3][0], dyW = dd[3][1];
        double nxN = nn[0][0], nyN = nn[0][1], nxE = nn[1][0], nyE = nn[1][1];
        double nxS = nn[2][0], nyS = nn[2][1], nxW = nn[3][0], nyW = nn[3][1];
        double D; double[4] fx, fy; double Ix, Iy, Cx = 0.0, Cy = 0.0;
        switch (wall) {
        case -1:
            D = mixin(R_D);
            fx = [mixin(R_Nx), mixin(R_Ex), mixin(R_Sx), mixin(R_Wx)]; Ix = mixin(R_Ix);
            fy = [mixin(R_Ny), mixin(R_Ey), mixin(R_Sy), mixin(R_Wy)]; Iy = mixin(R_Iy);
            break;
        case 0:
            D = mixin(ZGN_D);
            fx = [mixin(ZGN_Nx), mixin(ZGN_Ex), mixin(ZGN_Sx), mixin(ZGN_Wx)]; Ix = mixin(ZGN_Ix);
            fy = [mixin(ZGN_Ny), mixin(ZGN_Ey), mixin(ZGN_Sy), mixin(ZGN_Wy)]; Iy = mixin(ZGN_Iy);
            Cx = mixin(ZGN_Gx)/D; Cy = mixin(ZGN_Gy)/D;
            break;
        case 1:
            D = mixin(ZGE_D);
            fx = [mixin(ZGE_Nx), mixin(ZGE_Ex), mixin(ZGE_Sx), mixin(ZGE_Wx)]; Ix = mixin(ZGE_Ix);
            fy = [mixin(ZGE_Ny), mixin(ZGE_Ey), mixin(ZGE_Sy), mixin(ZGE_Wy)]; Iy = mixin(ZGE_Iy);
            Cx = mixin(ZGE_Gx)/D; Cy = mixin(ZGE_Gy)/D;
            break;
        case 2:
            D = mixin(ZGS_D);
            fx = [mixin(ZGS_Nx), mixin(ZGS_Ex), mixin(ZGS_Sx), mixin(ZGS_Wx)]; Ix = mixin(ZGS_Ix);
            fy = [mixin(ZGS_Ny), mixin(ZGS_Ey), mixin(ZGS_Sy), mixin(ZGS_Wy)]; Iy = mixin(ZGS_Iy);
            Cx = mixin(ZGS_Gx)/D; Cy = mixin(ZGS_Gy)/D;
            break;
        default:
            D = mixin(ZGW_D);
            fx = [mixin(ZGW_Nx), mixin(ZGW_Ex), mixin(ZGW_Sx), mixin(ZGW_Wx)]; Ix = mixin(ZGW_Ix);
            fy = [mixin(ZGW_Ny), mixin(ZGW_Ey), mixin(ZGW_Sy), mixin(ZGW_Wy)]; Iy = mixin(ZGW_Iy);
            Cx = mixin(ZGW_Gx)/D; Cy = mixin(ZGW_Gy)/D;
            break;
        }
        double scale = 0.0, worst = 0.0;
        foreach (j; 0 .. 4) { scale = fmax(scale, fabs(fx[j]/D)); scale = fmax(scale, fabs(fy[j]/D)); }
        double sx = 0.0, sy = 0.0;
        foreach (j; 0 .. 4) {
            if (j == wall) continue;
            worst = fmax(worst, fabs(Wg[0][j] - fx[j]/D)/scale);
            worst = fmax(worst, fabs(Wg[1][j] - fy[j]/D)/scale);
            sx += Wg[0][j]; sy += Wg[1][j];
        }
        worst = fmax(worst, fabs(-sx - Ix/D)/scale);
        worst = fmax(worst, fabs(-sy - Iy/D)/scale);
        if (wall >= 0) {
            worst = fmax(worst, fabs(Wg[0][wall] - Cx)/fmax(1.0, fabs(Cx)));
            worst = fmax(worst, fabs(Wg[1][wall] - Cy)/fmax(1.0, fabs(Cy)));
        }
        return worst;
    }

    // Exactness on quadratic-without-cross-terms fields in 3D, and the slope rows.
    double check3d(ref Random rng, int nwalls)
    {
        double[3][6] base = [[0,1,0],[1,0,0],[0,-1,0],[-1,0,0],[0,0,1],[0,0,-1]];
        double[3][MAXF] dd; double[3][MAXF] nn; bool[MAXF] sl;
        foreach (j; 0 .. 6) {
            foreach (a; 0 .. 3) dd[j][a] = 2.0e-3*(base[j][a] + uniform(-0.25, 0.25, rng));
            foreach (a; 0 .. 3) nn[j][a] = base[j][a];
            sl[j] = false;
        }
        // walls on the first nwalls of (north, top, east): an edge/corner cell for nwalls >= 2
        int[3] wl = [0, 4, 1];
        foreach (w; 0 .. nwalls) sl[wl[w]] = true;
        double[MAXF][3] Wg;
        assert(reconstructionWeights(3, dd, nn, sl, Wg));
        // phi = g0.x + sum h_a x_a^2 ; slope rows carry g.n (the field's true wall slope)
        double[3] g0 = [uniform(-1.0, 1.0, rng), uniform(-1.0, 1.0, rng), uniform(-1.0, 1.0, rng)];
        double[3] hh = [uniform(-50.0, 50.0, rng), uniform(-50.0, 50.0, rng), uniform(-50.0, 50.0, rng)];
        double worst = 0.0;
        foreach (a; 0 .. 3) {
            double ga = 0.0;
            foreach (j; 0 .. 6) {
                double r;
                if (sl[j]) {
                    r = g0[0]*nn[j][0] + g0[1]*nn[j][1] + g0[2]*nn[j][2];
                } else {
                    r = 0.0;
                    foreach (b; 0 .. 3) r += g0[b]*dd[j][b] + hh[b]*dd[j][b]*dd[j][b];
                }
                ga += Wg[a][j]*r;
            }
            worst = fmax(worst, fabs(ga - g0[a]));
        }
        return worst;
    }

    void main()
    {
        auto rng = Random(20261007);
        double w2 = 0.0;
        foreach (trial; 0 .. 500) foreach (wall; -1 .. 4) w2 = fmax(w2, check2d(rng, wall));
        writefln("2D generic vs efieldderivatives.d (R_* and ZG{N,E,S,W}, incl. C): worst rel. mismatch %.3e", w2);
        double w3 = 0.0;
        foreach (trial; 0 .. 500) foreach (nw; 0 .. 4) w3 = fmax(w3, check3d(rng, nw));
        writefln("3D exactness on g.x + sum h_a x_a^2 (0-3 walls, skewed):        worst abs. error  %.3e", w3);
        assert(w2 < 1.0e-9, "2D generic weights disagree with the closed forms");
        assert(w3 < 1.0e-9, "3D weights are not exact on the model field");
        writeln("efieldstencil: PASS");
    }
}
