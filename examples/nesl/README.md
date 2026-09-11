# NESL: Non-Equilibrium Stagnation Line

_Taine J. Rossini_

---

## 1.0 Formulation

`Nesl` is a transient quasi-1D time-marching stagnation line code for cylinders and spheres, built on the finite-volume-method (FVM) and an explicit Euler update equation. The code is shock-capturing and can be run time accurately or to accelerated steady-state with local time stepping (LTS).

The formulation reduces the governing equations to one-dimension by accounting for an additional geometric source term, governed by the transverse velocity gradient $\beta=\frac{\partial v}{\partial y}$. This can be solved for across the shock layer with its own governing equation,

$$\rho u \frac{\partial \beta}{\partial x} + \rho \beta^2 = -\frac{\partial^2 p}{\partial y^2} + \frac{\partial}{\partial x} \left( \mu \frac{\partial \beta}{\partial x} \right)$$

This is then solved using Picard iteration and a TDMA for the viscous case, and an implicit march for the inviscid one, with the boundary condition at the shock set by its local radius of curvature. It is these source terms that provide the mass relief throughout the shock layer that a strictly 1D formulation cannot capture.

---

## 2.0 Usage

This code can be used to study the stagnation line of a sphere or cylinder, given that you can supply the freestream conditions, an appropriate one or two-temperature gas model, and effective nose radius.

It could commonly be used to analyse,

1. The non-equilibrium state behind a blunt body.
2. Transient pitot probe startup.
3. Stagnation point convective and diffusive heat flux.
4. Shock standoff and shock radius of curvature.

`Nesl` was written in a combination of `Python` and `Cython`, building on the `gdtk` `C` API.

`Nesl condition builder` allows for the parallelisation of simulations, allowing for condition building and mapping. This is useful in determining the optimum scaling conditions if one does not want to rely on binary-scaling.

### 2.1 Installation

`Nesl` is installed via the provided `makefile`, which additionally requires the `equilibrium-c` library and builds a dedicated virtual environment from [`requirements.txt`](requirements.txt). See [`docs/install_guide.md`](docs/install_guide.md) for the full setup.

### 2.2 Running nesl

A user must interact with `nesl` and `nesl condition builder` through the CLI, where each is driven by a `.toml` configuration file,

```
nesl --job=job.toml
nesl_condition_builder --job=template.toml --bounds=bounds.toml --max_cpus=8
```

Every setting recognised by both programs is listed in [`docs/settings.md`](docs/settings.md). The utility classes, `Helper`, `Grid`, `Prep`, and `Plot`, can also be imported into `Python` directly,

```Python
from nesl_utils import *
```

### 2.3 Examples

Eleven test cases have been provided in the [`examples/`](examples/) directory.
