# CRBL: Chemically Reacting Boundary Layer

_Taine J. Rossini_

---

## 1.0 Formulation

`Crbl` is a steady 2D or axisymmetric space-marching chemically reacting laminar boundary-layer code, built on an implicit finite-difference-method (FDM).

We are able to implement a space-marching solution in a subsonic domain (the boundary-layer) by assuming that our flow is only influenced by its upstream. This implies that there are no perturbations downstream that can propagate disturbances upstream.

### 1.1 Boundary-Layer

The equations solved are those in transformed space, $\xi$ and $\eta$, relying on the Levy-Lees formulation, naturally clustering near the wall. By defining the initial conditions using the self-similar solution, we can march downstream to each station, where the Picard linearised boundary-layer equations are solved until convergence. Note that inter-species diffusion is set by a constant $Le=1$.

Note that `crbl` computes solutions using surface arc length, $x$, and local normal, $y$, coordinates. To simulate an arbitrary geometry, you must first convert your geometry into these arc-length coordinates before running the code, and reverse the transformation to map the output back to your original coordinate space. Axisymmetric users must also supply the solver with the radius at each $x$ location.

### 1.2 Viscous Interaction

A boundary-layer code is only useful in some circumstances, as often the boundary-layer interacts with the outer inviscid flow, particularly in supersonic and hypersonic flows. This is known as a viscous interaction, where the displacement thickness, $\delta^*$, causes a shock to form, actively compressing the boundary-layer. To account for this, the main solver was coupled with the tangent wedge method in an iterative loop to capture this effect.

---

## 2.0 Usage

This code can be used to simulate viscous phenomena over a large range of geometries. Given that you can supply the boundary-layer edge conditions, and an appropriate gas model, you can use the script. The code is only set up to handle one-temperature chemistry schemes.

It could commonly be used to determine,

1. The displacement thickness over a generic body (external, internal, planar, or axisymmetric).
    - Nosecones.
    - Nozzles.
    - Shock or acceleration tubes.
2. The viscous pressure interaction over a flat plate or wedge.
3. Key boundary-layer loads over an arbitrary geometry.
    - Shear stress.
    - Conductive and diffusive heat flux.

`Crbl` was written as a pair of `Python` classes, `BoundaryLayer` and `ViscousInteraction`, and builds on the `gdtk` `Python` API.

### 2.1 Installation

`Crbl` is installed via the provided `makefile`, which additionally requires the `equilibrium-c` library and builds a dedicated virtual environment from [`requirements.txt`](requirements.txt). See [`docs/install_guide.md`](docs/install_guide.md) for the full setup.

### 2.2 Running crbl

The configuration script sets fields on the global `config` object, calls `config.init_gas_model()`, and constructs exactly one `BoundaryLayer` or `ViscousInteraction` object. `Crbl` executes the script in its own namespace and runs whichever object was built, writing `loads.dat` and `output.vts` to disk. The same classes can also be imported directly into an ordinary Python script instead of using the CLI. Full details of `config` and both classes are in [`docs/settings.md`](docs/settings.md).

### 2.3 Examples

The [`examples/`](examples/) directory has four scripts,

- [`test_bl.py`](examples/test_bl.py), boundary-layer configuration script, run via CLI.
- [`test_vi.py`](examples/test_vi.py), viscous-interaction configuration script, run via CLI.
- [`test_bl_import.py`](examples/test_bl_import.py), the same boundary-layer case run by import.
- [`test_vi_import.py`](examples/test_vi_import.py), the same viscous-interaction case run by import.
