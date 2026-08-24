#!/usr/bin/env python3
"""
Phase 0 ground truth for the Path 2 insulator boundary condition.

Two jobs:

 1. Reverse-engineer, and then VERIFY against the shipped code, the linear system
    that efieldderivatives.d's ZGN/ZGE/ZGS/ZGW families solve. Those families were
    machine-generated from Maxima in 2022 and the source .max file is not in the
    tree, so the convention is recovered here and checked numerically against the
    mixin strings themselves.

 2. Derive the one extra coefficient each family needs to support a NON-ZERO
    prescribed normal slope  grad(phi).n = g_n  at the wall (the Hall insulator
    condition), and check the closed form against a brute-force solve.

Run:  python3 zngbc.py
"""
import re, itertools
import numpy as np

# ---------------------------------------------------------------- the shipped strings
DERIV = "../../src/lmr/efield/efieldderivatives.d"

def load_consts(path):
    src = open(path).read()
    out = {}
    for m in re.finditer(r'const string (\w+)\s*=\s*"(.*?)"\s*;', src, re.S):
        out[m.group(1)] = " ".join(m.group(2).split()).replace("^^", "**")
    return out

C = load_consts(DERIV)

def ev(name, env):
    return eval(C[name], {"__builtins__": {}}, env)

# ---------------------------------------------------------------- the model
# efieldderivatives.d docstring:
#   dP = dpdx*dx + dpdy*dy + dp2dx2*(dx^2 - dy^2)/2 + dp2dxdy*dx*dy
# Unknown vector u = (px, py, a, b).  Row for a neighbour at offset (dx,dy):
def nbr_row_interior(dx, dy):
    # dP = px*dx + py*dy + a*(dx^2 - dy^2)/2 + b*dx*dy   (verified exactly, k=1)
    return [dx, dy, 0.5*(dx*dx - dy*dy), dx*dy]

def nbr_row_zg(dx, dy):
    # The ZG* families use a DIFFERENT model: no cross term, dx^2 and dy^2 independent.
    # dP = px*dx + py*dy + dxx*dx^2 + dyy*dy^2      (verified exactly below)
    return [dx, dy, dx*dx, dy*dy]

def bc_row(nx, ny):
    return [nx, ny, 0.0, 0.0]

DIRS = ["N", "E", "S", "W"]

def geom(rng):
    """A random, deliberately non-orthogonal 5-point stencil + a wall normal."""
    g = {}
    base = {"N": (0.0, 1.0), "E": (1.0, 0.0), "S": (0.0, -1.0), "W": (-1.0, 0.0)}
    for d, (x, y) in base.items():
        g["dx" + d] = x + 0.25 * rng.standard_normal()
        g["dy" + d] = y + 0.25 * rng.standard_normal()
    th = rng.uniform(0, 2 * np.pi)
    for d in DIRS:                      # every family's normal, same value
        g["nx" + d], g["ny" + d] = np.cos(th), np.sin(th)
    return g

def system(g, wall):
    """Assemble M so that M u = rhs, with the WALL neighbour's row replaced by the BC row."""
    rows, tag = [], []
    for d in DIRS:
        if d == wall:
            rows.append(bc_row(g["nx" + d], g["ny" + d])); tag.append("g")
        else:
            rows.append(nbr_row_zg(g["dx" + d], g["dy" + d])); tag.append(d)
    return np.array(rows), tag

# ---------------------------------------------------------------- job 1: verify convention
def verify_interior():
    """The R_* family, for reference: 4 neighbours, cross-term model."""
    src = open(DERIV).read()
    expr = " ".join(src.split("double denominator(")[1].split("double D = ")[1]
                    .split(";")[0].split()).replace("^^", "**")
    rng, worst = np.random.default_rng(11), 0.0
    for _ in range(200):
        g = geom(rng)
        M = np.array([nbr_row_interior(g["dx"+d], g["dy"+d]) for d in DIRS])
        Dn = eval(expr, {"__builtins__": {}}, dict(g))
        Minv = np.linalg.inv(M)
        for comp, row in (("x", 0), ("y", 1)):
            theirs = np.array([ev(f"{d}{comp}", g)/Dn for d in DIRS])
            mine = Minv[row, :]
            worst = max(worst, np.max(abs(theirs-mine))/np.max(abs(theirs)))
    return worst

FAM = {"N": "ZGN", "E": "ZGE", "S": "ZGS", "W": "ZGW"}

def verify_zg():
    rng, worst = np.random.default_rng(7), 0.0
    for _ in range(400):
        g = geom(rng)
        for wall in DIRS:
            fam = FAM[wall]
            M, tag = system(g, wall)
            Minv = np.linalg.inv(M)
            D = ev(fam + "_D", g)
            for comp, row in (("x", 0), ("y", 1)):
                mine, theirs = [], []
                for j, d in enumerate(tag):
                    if d == "g":
                        continue
                    mine.append(Minv[row, j])
                    theirs.append(ev(f"{fam}_{d}{comp}", g)/D)
                mine.append(-sum(mine))
                theirs.append(ev(f"{fam}_I{comp}", g)/D)
                mine, theirs = np.array(mine), np.array(theirs)
                worst = max(worst, np.max(abs(theirs-mine))/max(1e-300, np.max(abs(theirs))))
    return worst

print("Job 1 - recover the ZG*/R linear-system conventions")
print(f"  interior R_* (cross-term model)     : worst rel. mismatch = {verify_interior():.3e}")
print(f"  wall ZG?_*  (dx^2,dy^2 model)       : worst rel. mismatch = {verify_zg():.3e}")

# ------------------------------------------------- job 2: the non-zero-slope sensitivity
#
# With grad(phi).n = g_n prescribed instead of 0, the rhs gains g_n in the BC slot, so
#
#     (px, py) = (px, py)|_{g_n=0}  +  g_n * Cvec,     Cvec = Minv[0:2, bc_column]
#
# Everything else in the assembly is unchanged: the shipped ZG* weights ARE the
# g_n = 0 part.  Only Cvec is new.

def cvec(g, wall):
    M, tag = system(g, wall)
    j = tag.index("g")
    return np.linalg.inv(M)[0:2, j]

print()
print("Job 2 - sensitivity of the reconstructed gradient to a prescribed wall slope g_n")
rng = np.random.default_rng(3)
worst_n, gammas = 0.0, []
for _ in range(500):
    g = geom(rng)
    for wall in DIRS:
        Cv = cvec(g, wall)
        n = np.array([g["nx"+wall], g["ny"+wall]])
        t = np.array([n[1], -n[0]])           # the code's tangent, efield.d
        worst_n = max(worst_n, abs(Cv @ n - 1.0))
        gammas.append(Cv @ t)
print(f"  identity  Cvec.n == 1 : worst deviation = {worst_n:.3e}   <- must be round-off")
print(f"  gamma = Cvec.t        : range [{min(gammas):+.3f}, {max(gammas):+.3f}] "
      f"(0 on a symmetric stencil)")

# --- closed form for Cvec, as new mixin strings ---------------------------------------
import sympy as sp

def emit(wall):
    fam = FAM[wall]
    others = [d for d in DIRS if d != wall]
    dx = {d: sp.Symbol("dx"+d) for d in others}
    dy = {d: sp.Symbol("dy"+d) for d in others}
    nx, ny = sp.Symbol("nx"+wall), sp.Symbol("ny"+wall)
    rows = {d: [dx[d], dy[d], dx[d]**2, dy[d]**2] for d in others}
    M = sp.Matrix([[nx, ny, 0, 0]] + [rows[d] for d in others])
    # reorder rows so the BC row sits where the code's DIRS order puts it -- the sign of
    # det(M) must match D, so build M in the code's own N,E,S,W row order:
    Mo = sp.Matrix([[nx, ny, 0, 0] if d == wall else rows[d] for d in DIRS])
    adj = Mo.adjugate()
    j = DIRS.index(wall)
    detM = sp.expand(Mo.det())
    # The shipped _D is  s*det(M)  with s = -1 for N,E and +1 for S,W (the generator's
    # row ordering).  The shipped weights carry the same s, so it cancels there -- but
    # our new coefficients must be written against the SHIPPED _D, hence the same s.
    Dshipped = sp.expand(sp.sympify(C[fam + "_D"].replace("^^", "**")))
    s = sp.simplify(Dshipped / detM)
    assert s in (1, -1), (fam, s)
    Gx = sp.expand(s * adj[0, j])             # Cx = Gx / D_shipped
    Gy = sp.expand(s * adj[1, j])
    return fam, Gx, Gy, detM, Mo

def dstr(e):
    return str(sp.expand(e)).replace("**", "^^")

print()
print("  closed form (Cx = <fam>_Gx / <fam>_D, Cy = <fam>_Gy / <fam>_D):")
out = {}
for wall in DIRS:
    fam, Gx, Gy, detM, Mo = emit(wall)
    out[fam] = (Gx, Gy)
    print(f"      {fam}_Gx = {sp.factor(Gx)}")
    print(f"      {fam}_Gy = {sp.factor(Gy)}")

# numeric cross-check of the closed forms against the brute-force inverse
rng, worst = np.random.default_rng(5), 0.0
for _ in range(300):
    g = geom(rng)
    for wall in DIRS:
        fam = FAM[wall]
        Gx, Gy = out[fam]
        D = ev(fam+"_D", g)
        cf = np.array([float(Gx.subs(g))/D, float(Gy.subs(g))/D])
        worst = max(worst, np.max(abs(cf - cvec(g, wall))))
print(f"  closed form vs brute-force inverse: worst abs diff = {worst:.3e}")
