# STRATOS: Shocked Thermochemical Relaxation

_Taine J. Rossini_

---

## 1.0 Formulation

`Stratos` is a steady 1D space-marching post-shock thermochemical relaxation code, built on a finite-volume-method (FVM) and an explicit Euler update equation.

As in any initial-value-problem, we can supply the initial conditions using the Rankine-Hugoniot jump conditions, and march downstream to obtain our solution.

We are able to implement a space-marching solution in a subsonic domain by assuming that our flow is only influenced by its upstream. This implies that there are no bodies downstream that can propagate disturbances upstream, and that the flow is inviscid.

### 1.1 Normal Shock

By defining the initial conditions, it is as simple as marching downstream, and then updating our primitive variables using a Secant solver on the incoming fluxes. An interesting result of excluding the transient term is that we are now working at cell faces, unlike more traditional FVM at cell centres.

Since total enthalpy, mass, and momentum are conserved along a 1D streamline, we only need to track species, $\rho_i$, and our vibro-electronic energy, $e_{ve}$, for two-temperature flows.

### 1.2 Oblique Shock

Non-equilibrium oblique shocks are curved over a fixed wedge angle, which makes it inherently impossible to simulate these flows in 1D. However, by assuming the shock is locally straight, we can solve for an approximate solution.

Oblique shock theory tells us that the tangential component of velocity is conserved, and thus given a shock angle, $\beta$, we can march in the normal direction only, and get transported in the tangential direction. This inherently acts like a streamline tracer, and given a fixed shock angle, will output the curvature of the streamline due to non-equilibrium effects.

However, we can extend this such that we can evaluate to what degree our flow is in non-equilibrium, based on the fact that $\beta_{eq} \leq \beta_{non-eq} \leq \beta_{frozen}$. Again, from oblique shock theory, we know that the velocity vector of the flow must be parallel to the body. We can then iterate on $\beta$ such that the final velocity vector in our domain is parallel to the wedge.

---

## 2.0 Usage

This code can be used to study the thermochemical relaxation behind oblique and normal shocks. Given that you can supply the freestream conditions, and an appropriate one or two-temperature gas model, you can use the script.

It could commonly be used to determine,

1. The non-equilibrium state behind a moving normal shock, such as is often experienced in UQ's impulse facilities.
2. The path of a streamline behind a curved shock.
3. The degree of non-equilibrium in a wedge flow.

`Stratos` was written as a `Python` class and builds on the `gdtk` `Python` API.

### 2.1 Installation

`Stratos` is installed via the provided `makefile`, which additionally requires `numpy`, `scipy`, `cffi`, and `matplotlib` in the local `Python` environment. See [`docs/install_guide.md`](docs/install_guide.md) for the full setup.

### 2.2 Running stratos

`Stratos` has no CLI; it is a single class imported into an ordinary Python script. Construct one `stratos` object per freestream, then call `normal()`, `oblique()`, or `shock_angle()` as required. Nothing is written to disk or plotted, each call returns a dictionary (or tuple) for the script to use directly. Full details of the class are in [`docs/settings.md`](docs/settings.md).

### 2.3 Examples

The [`examples/`](examples/) directory has four worked job scripts,

- [`1T_normal`](examples/1T_normal), one-temperature normal shock relaxation, validated against Marrone and Poshax reference data.
- [`2T_normal`](examples/2T_normal), two-temperature normal shock relaxation, validated against Eilmer.
- [`wedge`](examples/wedge), oblique shock sweep over a range of wedge angles, plotting temperature and density along the body.
- [`reconstruct_density`](examples/reconstruct_density), uses `shock_angle()` along a measured shock/body shape to reconstruct the density field around a body.
