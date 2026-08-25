"""Is Parent's minmod anti-diffusion usable as a deferred correction here?

The correction is nonlinear, so it has to be iterated. Its per-sweep gain is what decides
whether it is a practical fix or a source of limit cycles. Measure that directly, with
under-relaxation, on the two cases that matter: a smooth varying-sigma channel (where the
first-order error is small) and the cold electrode wall (where it dominates).
"""
import numpy as np
from channel import Grid, solve

def sweep(g, sig, beta, walls, drop, relax, iters=60, tag=""):
    kw = dict(walls=walls, drop=drop, drop_fix="vertex", robin_bc=True, uxb_at_wall=True)
    phi, ex = solve(g, sig, beta, 10000.0, 0.5, conv="upwind", **kw)
    scale = max(1e-30, np.max(np.abs(ex)))
    errs = [np.max(np.abs(phi - ex))/scale]
    dphis = []
    for _ in range(iters):
        pn, _ = solve(g, sig, beta, 10000.0, 0.5, conv="antidiff", phi_lag=phi, **kw)
        pn = relax*pn + (1.0 - relax)*phi
        dphis.append(np.max(np.abs(pn - phi))/scale)
        phi = pn
        errs.append(np.max(np.abs(phi - ex))/scale)
        if not np.isfinite(errs[-1]) or errs[-1] > 1e6:
            break
    gain = (dphis[-1]/dphis[-2]) if len(dphis) > 2 and dphis[-2] > 0 else float("nan")
    print(f"  {tag:<34} relax={relax:4.2f}  err {errs[0]:9.3e} -> {errs[-1]:9.3e}"
          f"   sweeps {len(dphis):3d}  last gain {gain:8.3f}")
    return errs

smooth = lambda x, y: 300.0*(1.0 + 0.5*np.sin(30.0*x)*np.cos(60.0*y))
cold   = lambda x, y: 300.0*(0.001 + np.tanh(400.0*min(y, 0.03 - y))**2)

print("Smooth sigma, insulating ends (first-order error is already small)")
for r in (1.0, 0.5, 0.2, 0.05):
    sweep(Grid(24, 24), smooth, 15.0, "EW", False, r, tag="smooth, beta=15")

print("\nCold (sigma-collapsing) electrode walls -- the case that actually fails")
for r in (1.0, 0.5, 0.2, 0.05, 0.02):
    sweep(Grid(24, 24), cold, 15.0, "EW", True, r, tag="cold wall, beta=15")
