#!/usr/bin/env python3
"""
Phase 0 accuracy check for the Path 2 insulator boundary condition.

A standalone, solver-independent re-implementation of efield.d's UPWIND (Path 2)
assembly on a structured 2-D block, run against an EXACT solution, so the boundary
condition can be judged on accuracy rather than on the round-off conservation check
(which only proves the operator is conservative, not that it is right).

The exact solution
------------------
Uniform u and uniform B = Bz zhat.  Take phi = -(u x B) is a constant vector field,
so phi_exact with grad(phi) = (u x B) makes E* = -grad(phi) + u x B = 0 and therefore
J = sigma_t E* = 0 EVERYWHERE, for ANY sigma and ANY beta.  With u = (ux, 0):

    u x B = (0, -ux*Bz),      phi_exact = -ux*Bz*y

Every face carries zero current, so J.n = 0 on the insulating walls is satisfied by
the exact field: it is a legitimate solution of the insulated-wall problem, and any
consistent scheme must reproduce it.  The grid is SHEARED so the walls are not
aligned with a coordinate direction -- that makes  d(phi)/dt  non-zero along the
wall, which is precisely the term the shipped BC drops.

Boundaries: north and south are ZeroNormalGradient (insulating walls); east and west
use known-phi ghost cells, assembled through the interior branch exactly as a
block-shared face is.

Run:  python3 channel.py
"""
import re
import numpy as np

DERIV = "../../src/lmr/efield/efieldderivatives.d"
SRC = open(DERIV).read()
CONST = {m.group(1): " ".join(m.group(2).split()).replace("^^", "**")
         for m in re.finditer(r'const string (\w+)\s*=\s*"(.*?)"\s*;', SRC, re.S)}
DENOM = " ".join(SRC.split("double denominator(")[1].split("double D = ")[1]
                 .split(";")[0].split()).replace("^^", "**")

# g_n sensitivity coefficients, derived and verified in zngbc.py:  C = (Gx/D, Gy/D)
GX = {
 "ZGN": "-dxE**2*dyS**2*dyW + dxE**2*dyS*dyW**2 + dxS**2*dyE**2*dyW - dxS**2*dyE*dyW**2 - dxW**2*dyE**2*dyS + dxW**2*dyE*dyS**2",
 "ZGE": "dxN**2*dyS**2*dyW - dxN**2*dyS*dyW**2 - dxS**2*dyN**2*dyW + dxS**2*dyN*dyW**2 + dxW**2*dyN**2*dyS - dxW**2*dyN*dyS**2",
 "ZGS": "-dxE**2*dyN**2*dyW + dxE**2*dyN*dyW**2 + dxN**2*dyE**2*dyW - dxN**2*dyE*dyW**2 - dxW**2*dyE**2*dyN + dxW**2*dyE*dyN**2",
 "ZGW": "dxE**2*dyN**2*dyS - dxE**2*dyN*dyS**2 - dxN**2*dyE**2*dyS + dxN**2*dyE*dyS**2 + dxS**2*dyE**2*dyN - dxS**2*dyE*dyN**2",
}
GY = {
 "ZGN": "-dxE**2*dxS*dyW**2 + dxE**2*dxW*dyS**2 + dxE*dxS**2*dyW**2 - dxE*dxW**2*dyS**2 - dxS**2*dxW*dyE**2 + dxS*dxW**2*dyE**2",
 "ZGE": "dxN**2*dxS*dyW**2 - dxN**2*dxW*dyS**2 - dxN*dxS**2*dyW**2 + dxN*dxW**2*dyS**2 + dxS**2*dxW*dyN**2 - dxS*dxW**2*dyN**2",
 "ZGS": "-dxE**2*dxN*dyW**2 + dxE**2*dxW*dyN**2 + dxE*dxN**2*dyW**2 - dxE*dxW**2*dyN**2 - dxN**2*dxW*dyE**2 + dxN*dxW**2*dyE**2",
 "ZGW": "dxE**2*dxN*dyS**2 - dxE**2*dxS*dyN**2 - dxE*dxN**2*dyS**2 + dxE*dxS**2*dyN**2 + dxN**2*dxS*dyE**2 - dxN*dxS**2*dyE**2",
}

DIRS = ["N", "E", "S", "W"]          # efield.d's cell.iface order
FAM = {"N": "ZGN", "E": "ZGE", "S": "ZGS", "W": "ZGW"}


def weights(env, wall):
    """Return (fdx[4], fdy[4], Ix, Iy, D, Cx, Cy) exactly as efield.d builds them."""
    if wall is None:
        D = eval(DENOM, {"__builtins__": {}}, env)
        fdx = [eval(CONST[d + "x"], {"__builtins__": {}}, env) for d in DIRS]
        fdy = [eval(CONST[d + "y"], {"__builtins__": {}}, env) for d in DIRS]
        Ix = eval(CONST["Ix"], {"__builtins__": {}}, env)
        Iy = eval(CONST["Iy"], {"__builtins__": {}}, env)
        return fdx, fdy, Ix, Iy, D, 0.0, 0.0
    f = FAM[wall]
    D = eval(CONST[f + "_D"], {"__builtins__": {}}, env)
    fdx = [eval(CONST[f + "_" + d + "x"], {"__builtins__": {}}, env) for d in DIRS]
    fdy = [eval(CONST[f + "_" + d + "y"], {"__builtins__": {}}, env) for d in DIRS]
    Ix = eval(CONST[f + "_Ix"], {"__builtins__": {}}, env)
    Iy = eval(CONST[f + "_Iy"], {"__builtins__": {}}, env)
    Cx = eval(GX[f], {"__builtins__": {}}, env) / D
    Cy = eval(GY[f], {"__builtins__": {}}, env) / D
    return fdx, fdy, Ix, Iy, D, Cx, Cy


class Grid:
    def __init__(self, nx, ny, Lx=0.10, H=0.030, shear=0.35, stretch=1.6):
        self.nx, self.ny = nx, ny
        i = np.arange(nx + 1) / nx
        j = np.arange(ny + 1) / ny
        xs = Lx * i
        # tanh clustering towards both walls, then shear the whole block
        s = np.tanh(stretch * (2 * j - 1)) / np.tanh(stretch)
        ys = H * 0.5 * (1 + s)
        X, Y = np.meshgrid(xs, ys, indexing="ij")       # (nx+1, ny+1)
        self.vx = X
        self.vy = Y + shear * X
        # cell centres
        self.cx = 0.25 * (self.vx[:-1, :-1] + self.vx[1:, :-1] + self.vx[:-1, 1:] + self.vx[1:, 1:])
        self.cy = 0.25 * (self.vy[:-1, :-1] + self.vy[1:, :-1] + self.vy[:-1, 1:] + self.vy[1:, 1:])

    def face(self, i, j, d):
        """Face centre, outward unit normal, length, and its two vertex indices."""
        if d == "N":   a, b = (i, j + 1), (i + 1, j + 1)
        elif d == "S": a, b = (i + 1, j), (i, j)
        elif d == "E": a, b = (i + 1, j + 1), (i + 1, j)
        elif d == "W": a, b = (i, j), (i, j + 1)
        ax, ay = self.vx[a], self.vy[a]
        bx, by = self.vx[b], self.vy[b]
        ex, ey = bx - ax, by - ay
        S = np.hypot(ex, ey)
        n = np.array([ey, -ex]) / S            # outward for the ordering above
        return np.array([0.5 * (ax + bx), 0.5 * (ay + by)]), n, S, (a, b)


def solve(g, sigma_fn, beta, ux, Bz, robin_bc, uxb_at_wall, walls="NS", tang="central",
          drop=False, drop_fix="vertex", conv="upwind"):
    """Assemble and solve exactly as efield.d's upwind branch does.

    robin_bc     : apply the derived Robin slope g_n at ZNG walls (False = shipped grad.n=0)
    uxb_at_wall  : apply the u x B source at ZNG faces (False = shipped, dropped)
    """
    nx, ny = g.nx, g.ny
    N = nx * ny
    kid = lambda i, j: j * nx + i
    uxB = np.array([0.0, -ux * Bz])

    # sigma_H at vertices: cell values scattered and averaged (computeHallVertexField)
    sH_c = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            sH_c[i, j] = sigma_fn(g.cx[i, j], g.cy[i, j]) * beta / (1 + beta * beta)
    sH_v = np.zeros((nx + 1, ny + 1)); cnt = np.zeros((nx + 1, ny + 1))
    for i in range(nx):
        for j in range(ny):
            for a in [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]:
                sH_v[a] += sH_c[i, j]; cnt[a] += 1
    sH_v /= cnt

    A = np.zeros((N, N)); rhs = np.zeros(N)

    def ghost_phi(i, j, d):
        """Known-phi ghost beyond an east/west face (assembled as a shared face)."""
        p, n, S, _ = g.face(i, j, d)
        c = np.array([g.cx[i, j], g.cy[i, j]])
        q = c + 2.0 * ((p - c) @ n) * n
        return q, -ux * Bz * q[1]

    for i in range(nx):
        for j in range(ny):
            k = kid(i, j)
            c = np.array([g.cx[i, j], g.cy[i, j]])
            wall = None
            info = {}
            for d in DIRS:
                p, n, S, vtx = g.face(i, j, d)
                edge = ((d == "N" and j == ny - 1) or (d == "S" and j == 0) or
                        (d == "E" and i == nx - 1) or (d == "W" and i == 0))
                if edge and d in walls:        kind, other, phin = "zng", p, None
                elif edge and drop:            kind, other, phin = "drop", p, None
                elif edge:                     kind, (other, phin) = "ghost", ghost_phi(i, j, d)
                else:
                    kind, phin = "int", None
                    oi, oj = {"N": (i, j + 1), "S": (i, j - 1), "E": (i + 1, j), "W": (i - 1, j)}[d]
                    other = np.array([g.cx[oi, oj], g.cy[oi, oj]])
                    info[d] = kid(oi, oj)
                if kind == "zng":
                    wall = d
                info.setdefault(d, None)
                info[d + "_kind"] = kind
                info[d + "_p"] = p; info[d + "_n"] = n; info[d + "_S"] = S
                info[d + "_o"] = other; info[d + "_phin"] = phin; info[d + "_v"] = vtx

            env = {}
            for d in DIRS:
                env["dx" + d] = info[d + "_o"][0] - c[0]
                env["dy" + d] = info[d + "_o"][1] - c[1]
                env["nx" + d] = info[d + "_n"][0]
                env["ny" + d] = info[d + "_n"][1]
            fdx, fdy, Ix, Iy, D, Cx, Cy = weights(env, wall)

            # ---- gradient weights: grad(phi) = sum_j Fx[j]*phi_j  (+ const) -----------
            # column layout: 0..3 = N,E,S,W neighbours, 4 = self
            Fx = np.array([fdx[0], fdx[1], fdx[2], fdx[3], Ix]) / D
            Fy = np.array([fdy[0], fdy[1], fdy[2], fdy[3], Iy]) / D
            gx_const = gy_const = 0.0

            if wall is not None and robin_bc:
                n_w = info[wall + "_n"]
                t_w = np.array([n_w[1], -n_w[0]])          # efield.d's tangent
                uw = np.array([ux, 0.0])                   # face velocity (uniform here)
                uxB_f = np.array([uw[1] * Bz, -uw[0] * Bz])
                s_bc = uxB_f @ n_w + beta * (uxB_f @ t_w)  # (uxB).n + beta*(uxB).t
                if tang == "central":
                    #   g_n = (s_bc - beta*t.grad_h)/(1 + beta*gamma),  gamma = C.t
                    gamma = Cx * t_w[0] + Cy * t_w[1]
                    den = 1.0 + beta * gamma
                    tau = t_w[0] * Fx + t_w[1] * Fy
                    Fx = Fx - beta * Cx * tau / den
                    Fy = Fy - beta * Cy * tau / den
                    gx_const, gy_const = Cx * s_bc / den, Cy * s_bc / den
                else:
                    # UPWIND the tangential derivative against ONE along-wall
                    # neighbour, chosen so this term's diagonal contribution is
                    # stabilising (sign(d.t) = -sign(beta)). A one-sided difference is
                    # exact on a linear field, so no accuracy is given up -- provided the
                    # neighbour's NORMAL offset is accounted for, which on a skewed or
                    # sheared grid it must be:
                    #     phi_j - phi_k = p*(t.grad) + q*(n.grad),  p = d.t, q = d.n
                    #     n.grad = g_n, so   t.grad = (Delta - q*g_n)/p
                    #     g_n = s_bc - beta*t.grad   =>   g_n = (p*s_bc - beta*Delta)/(p - beta*q)
                    best = None
                    for m, dd in enumerate(DIRS):
                        if info[dd + "_kind"] == "zng":
                            continue
                        dvec = info[dd + "_o"] - c
                        pp = dvec @ t_w
                        if abs(pp) < 1e-14 or beta * pp > 0.0:
                            continue
                        if best is None or abs(pp) < abs(best[1]):
                            best = (m, pp, dvec @ n_w)
                    den = None
                    if best is not None:
                        m, pp, qq = best
                        den = pp - beta * qq
                        if abs(den) < 0.2 * abs(pp):
                            den = None
                    if den is None:                        # fall back on the reconstruction
                        gamma = Cx * t_w[0] + Cy * t_w[1]
                        d2 = 1.0 + beta * gamma
                        tau = t_w[0] * Fx + t_w[1] * Fy
                        Fx = Fx - beta * Cx * tau / d2
                        Fy = Fy - beta * Cy * tau / d2
                        gx_const, gy_const = Cx * s_bc / d2, Cy * s_bc / d2
                    else:
                        Fx = Fx.copy(); Fy = Fy.copy()
                        Fx[m] += Cx * (-beta / den); Fx[4] += Cx * (beta / den)
                        Fy[m] += Cy * (-beta / den); Fy[4] += Cy * (beta / den)
                        gx_const, gy_const = Cx * pp * s_bc / den, Cy * pp * s_bc / den

            # ---- face loop -----------------------------------------------------------
            for d in DIRS:
                kind = info[d + "_kind"]
                n = info[d + "_n"]; S = info[d + "_S"]; p = info[d + "_p"]
                sigF = sigma_fn(p[0], p[1])
                sigP = sigF / (1 + beta * beta)
                e = info[d + "_o"] - c
                emag = np.hypot(*e); ehat = e / emag

                if kind == "drop":
                    # A prescribed-flux face (sheath electrode, or the insulator strip
                    # between segments). Zeroing its stencil removes the PEDERSEN flux and
                    # the source, but under Path 2 the face's HALL flux is not a local
                    # term at all -- it lives in the contour sum -S_an*phi, which must not
                    # be gated or the telescoping breaks. So subtract the face's own
                    # sigma_H*(t.grad phi) explicitly instead: that leaves
                    #     sum over the OTHER faces of the full physical flux,
                    # is zero for a uniform phi (so the conservation check still holds),
                    # and is one-sided only at a domain boundary, where there is no second
                    # cell for it to be conservative with.
                    # sigma_H must be taken on the SAME basis as the convection term
                    # (vertex averages of the CELL values). Using the face's own gas state
                    # differs wherever sigma varies sharply across the wall -- a cold
                    # electrode face has sigma ~ 0 while its vertices carry the hot
                    # near-wall cell value -- and then the subtraction cancels nothing.
                    if drop_fix == "vertex":
                        va_, vb_ = info[d + "_v"]
                        sH = 0.5 * (sH_v[va_] + sH_v[vb_])
                    elif drop_fix == "face":
                        sH = sigF * beta / (1 + beta * beta)
                    else:
                        sH = 0.0
                    sfacx, sfacy, sfac = -sH * n[1], sH * n[0], 0.0
                elif kind == "zng":
                    sfacx, sfacy, sfac = sigP * n[0], sigP * n[1], 0.0
                else:
                    mS = sigP * n
                    mdote = ehat @ mS
                    sfacx, sfacy = mS[0] - ehat[0] * mdote, mS[1] - ehat[1] * mdote
                    sfac = mdote / emag
                    # direct (hybrid) part
                    A[k, k] += -S * sfac
                    if kind == "int":
                        A[k, info[d]] += S * sfac
                    else:
                        rhs[k] -= S * sfac * info[d + "_phin"]

                # FD-stencil part (all faces share the cell's Fx/Fy)
                cols = [info[dd] for dd in DIRS] + [k]
                kinds = [info[dd + "_kind"] for dd in DIRS] + ["self"]
                phis = [info[dd + "_phin"] for dd in DIRS] + [None]
                for m, col in enumerate(cols):
                    w = S * (sfacx * Fx[m] + sfacy * Fy[m])
                    if w == 0.0:
                        continue
                    if kinds[m] == "zng":
                        continue                       # ZNG supplies no phi data point
                    if kinds[m] == "drop":
                        dm = info[DIRS[m] + "_o"] - c
                        dopp = info[DIRS[(m + 2) % 4] + "_o"] - c
                        tt = (dm @ dopp) / (dopp @ dopp)
                        A[k, k] += (1.0 - tt) * w
                        oc = info[DIRS[(m + 2) % 4]]
                        if oc is not None:
                            A[k, oc] += tt * w
                        continue
                    if kinds[m] == "ghost":
                        rhs[k] -= w * phis[m]
                    else:
                        A[k, col] += w
                rhs[k] -= S * (sfacx * gx_const + sfacy * gy_const)

                # Path 2 Hall convection:  -S_an * phi_donor
                t_f = np.array([n[1], -n[0]])
                (va, vb) = info[d + "_v"]
                dd = np.array([g.vx[vb] - g.vx[va], g.vy[vb] - g.vy[va]])
                along = dd @ t_f
                s0, s1 = sH_v[va], sH_v[vb]
                S_an = (s1 - s0) if along >= 0 else (s0 - s1)
                if S_an != 0.0:
                    interior = kind in ("int", "ghost")
                    if conv == "central" and kind == "int":
                        A[k, k] += -0.5 * S_an
                        A[k, info[d]] += -0.5 * S_an
                    elif conv == "central" and kind == "ghost":
                        A[k, k] += -0.5 * S_an
                        rhs[k] -= -0.5 * S_an * info[d + "_phin"]
                    elif S_an > 0.0 or not interior:
                        A[k, k] += -S_an
                    elif kind == "int":
                        A[k, info[d]] += -S_an
                    else:
                        rhs[k] -= -S_an * info[d + "_phin"]

                # u x B source, with the full tensor  m = sigma_P*(n + beta*t)
                if kind == "drop":
                    pass
                elif kind != "zng" or uxb_at_wall:
                    m = sigP * (n + beta * t_f)
                    rhs[k] += S * (m @ uxB)

    # gauge: pin one cell (all-Neumann in y, ghosts fix x, but keep it well posed)
    k0 = kid(nx // 2, ny // 2)
    A[k0, :] = 0.0; A[k0, k0] = 1.0
    rhs[k0] = -ux * Bz * g.cy[nx // 2, ny // 2]

    phi = np.linalg.solve(A, rhs)
    exact = (-ux * Bz * g.cy).reshape(-1, order="F")
    return phi, exact


def run(label, sigma_fn, betas, ns=(12, 24, 48), walls="NS", tang="central", drop=False,
        drop_fix="vertex", conv="upwind"):
    print(f"\n{label}")
    print(f"    {'beta':>6} {'N':>5} | {'shipped BC':>12} {'rate':>6} | {'Robin BC':>12} {'rate':>6}")
    for beta in betas:
        prev = {}
        for n in ns:
            g = Grid(n, n)
            row = []
            for tag, kw in (("old", dict(robin_bc=False, uxb_at_wall=False)),
                            ("new", dict(robin_bc=True, uxb_at_wall=True))):
                phi, ex = solve(g, sigma_fn, beta, 10000.0, 0.5, walls=walls, tang=tang, drop=drop, drop_fix=drop_fix, conv=conv, **kw)
                err = np.max(np.abs(phi - ex)) / max(1e-30, np.max(np.abs(ex)))
                rate = np.log2(prev[tag] / err) if tag in prev else float("nan")
                prev[tag] = err
                row += [err, rate]
            print(f"    {beta:6.1f} {n:5d} | {row[0]:12.3e} {row[1]:6.2f} | {row[2]:12.3e} {row[3]:6.2f}")


if __name__ == "__main__":
    print("Exact solution phi = -ux*Bz*y  (E* = 0, so J = 0 and every wall is insulating)")
    run("Test A - uniform sigma (sigma_H uniform: the Path 2 convection term is identically zero)",
        lambda x, y: 300.0, betas=(0.0, 1.2, 15.0))
    run("Test B - varying sigma (Path 2 convection active; first-order upwind, expect rate ~1)",
        lambda x, y: 300.0 * (1.0 + 0.5 * np.sin(30.0 * x) * np.cos(60.0 * y)),
        betas=(1.2, 15.0))
    # C6_pow's ZeroNormalGradient faces are the axial ENDS (inflow/outflow), not the
    # channel walls -- those are SheathField electrodes. At an end, t is the TRANSVERSE
    # direction, so d(phi)/dt is the full Faraday field and beta*(uxB).t ~ beta*u*B: both
    # new terms are at their largest. This exercises the ZGE/ZGW families.
    run("Test C - insulating ENDS (the C6_pow configuration; ZGE/ZGW families)",
        lambda x, y: 300.0, betas=(1.2, 15.0), walls="EW")
    # Test E is the actual C6_pow corner: an insulating END meeting a wall that efield.d
    # treats the SheathField way -- flux dropped, data point extrapolated. This is where
    # the solver put its 4e7 V/m spike.
    run("Test E - insulating ENDS + dropped-flux walls, AS SHIPPED (the C6_pow corner)",
        lambda x, y: 300.0, betas=(1.2, 15.0), walls="EW", drop=True, drop_fix=None)
    run("Test F - same, with the dropped face's sigma_H*(t.grad phi) subtracted",
        lambda x, y: 300.0, betas=(1.2, 15.0), walls="EW", drop=True, drop_fix="vertex")
    # A cold electrode: sigma collapses in the last fraction of a millimetre, so the FACE
    # value and the vertex average of the CELL values disagree by orders of magnitude.
    cold = lambda x, y: 300.0 * (0.001 + np.tanh(400.0 * min(y, 0.03 - y)) ** 2)
    run("Test G - cold (sigma-collapsing) electrode walls, sigma_H taken at the FACE",
        cold, betas=(15.0,), walls="EW", drop=True, drop_fix="face")
    run("Test G' - the same, sigma_H taken from the VERTEX field (matches convection)",
        cold, betas=(15.0,), walls="EW", drop=True, drop_fix="vertex")
    # Is the residual error the DONOR-CELL upwinding (Parent's terms 1-3 without the
    # minmod anti-diffusion of term 4)? Re-run the same case with a second-order central
    # face value in the convection term. Central is not monotone -- it is a diagnostic
    # here, not a proposal -- but it isolates first-order upwind error from everything
    # else.
    run("Test G'' - vertex sigma_H, CENTRAL face value in the convection term",
        cold, betas=(15.0,), walls="EW", drop=True, drop_fix="vertex", conv="central")
    run("Test D - insulating ENDS, varying sigma",
        lambda x, y: 300.0 * (1.0 + 0.5 * np.sin(30.0 * x) * np.cos(60.0 * y)),
        betas=(15.0,), walls="EW")
