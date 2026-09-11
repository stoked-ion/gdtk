# NESL and NCB: Reference Manual

_Taine J. Rossini_

Version 1.0.0.

---

## Abstract

This document lists every setting recognised by `nesl`, the stagnation-line solver, and `nesl_condition_builder`, hereafter `ncb`, the parallel condition builder. Anything not listed here is ignored.

---

## Command line

### NESL

```
nesl [option]

    --help          Displays the help message and exits.
    --job=<file>    Runs a simulation using a .toml configuration file.
    --grid=<file>   Builds the grid from a .toml configuration file and exits.
```

One option is given per run.

- **`--job=<file>`** runs the simulation described by the configuration file.
- **`--grid=<file>`** builds the grid from the same configuration file, writes `grid.png` and `grid.vtk`, and quits without solving the flow.
- Running with no argument, or with `--help`, prints the banner and exits.

### NCB

```
nesl_condition_builder [options]

    --help          Displays the help message and exits.
    --job=<file>    Template .toml configuration file.
    --bounds=<file> Bounds to test, specified using a .toml configuration file.
    --max_cpus=<N>  Number of cores to spread the condition building across.
```

Arguments are read **by position**, not by name.

- **`--job=<template.toml>`** (first, required) must be a valid NESL job file.
- **`--bounds=<ncb.toml>`** (second, required) is the bounds file described under [NCB bounds file settings](#ncb-bounds-file-settings).
- **`--max_cpus=<N>`** (third, optional) defaults to one process.

---

## NESL job file settings

### Required settings

The run aborts with a message if any of these is missing.

| Key | Type | Description |
| --- | --- | --- |
| `GAS_MODEL` | string | Path to the gas-model file. |
| `MASS_FRAC` | dict | Freestream mass fractions, keyed by species name. |
| `T_INF` | float | Freestream temperature, K. |
| `U_INF` | float | Freestream velocity, m/s. |
| `RADIUS` | float | Body nose radius, m. |
| `DOMAIN` | float | Length of the stagnation-line domain, m. |
| `CELLS` | int | Number of cells in the grid. |

In addition, exactly one of `P_INF` or `RHO_INF` must be supplied; the run aborts if neither is present.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `P_INF` | float | `None` | Freestream pressure, Pa. |
| `RHO_INF` | float | `None` | Freestream density, kg/m$^3$. |

If both are given, `P_INF` is used.

#### Species-keyed dictionaries

`MASS_FRAC`, `WALL_MASS_FRAC` and `RECOM_RATE` are all dictionaries keyed by species name. Names must match the gas model exactly, and *any species absent from the dictionary is set to zero*.

```toml
MASS_FRAC = {N2 = 0.767, O2 = 0.233}
```

### Thermochemistry

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `CHEM_MODEL` | string | *absent* | Path to the reaction file. Chemistry is on when this is given and off when it is not. |
| `EXCHANGE_MODEL` | string | `""` | Path to the energy-exchange file, for two-temperature runs. |
| `T_FROZEN` | float | `1.0` | Cells below this temperature are frozen and no chemistry occurs. |

### Geometry and wall

| Key | Type / values | Default | Description |
| --- | --- | --- | --- |
| `AXISYMMETRIC` | bool | `true` | `true` for a sphere, `false` for a cylinder. |
| `CURVATURE` | string | `"empirical"` | Method to determine radius of curvature of the shock. |
| `T_WALL` | float | `-1.0` | Isothermal wall temperature, K. Only takes effect with `VISCOSITY` on; if `VISCOSITY` is on and `T_WALL` is not set, the wall is adiabatic. |

#### Curvature models

- **`"empirical"`** Uses an empirical correlation to determine, $R_s$.
- **`"thin_shock_layer"`** Relies on the simple approximation, $R_s = R_b + \Delta$. Can under-predict shock standoff for lower density ratios.

### Viscous and wall-catalysis settings

| Key | Type / values | Default | Description |
| --- | --- | --- | --- |
| `VISCOSITY` | bool | `false` | Turns on viscous effects. |
| `SENSOR` | bool | `true` | Jameson pressure sensor that damps the viscous fluxes near the shock. Improves stability on startup, but valid for steady runs only. |
| `BETA_SMOOTHING` | bool | `true` | Applies smoothing to $\beta$ across the shock to account for a finite shock thickness. |
| `LEWIS_NUM` | float | `1.0` | Lewis number used for species diffusion. |
| `CATALYTIC` | string | `"non_catalytic"` | Wall catalysis model. |
| `WALL_MASS_FRAC` | dict | empty | Wall mass fractions, keyed by species. Used only by `"fixed_composition"`. |
| `RECOM_RATE` | dict | *absent* | Per-species recombination data. Used only by `"finite_rate"`. |

#### Catalysis models

- **`"non_catalytic"`** Fixes a zero species gradient across the wall.
- **`"finite_rate"`** Each species recombines at the wall at a rate set by its
  efficiency, defined in `RECOM_RATE`.
- **`"fixed_composition"`** The wall composition is held at `WALL_MASS_FRAC`.
- **`"equilibrium"`** The wall composition is held at the equilibrium
  composition for the stagnation pressure and wall temperature.

#### `RECOM_RATE` format

Each catalytic species maps to a dictionary carrying an efficiency and the species the recombined mass is deposited into.

```toml
RECOM_RATE = {O = {efficiency = 0.01, product = "O2"}, N = {efficiency = 0.005, product = "N2"}}
```

- **`efficiency`** (float) is the recombination efficiency used in calculating the catalytic velocity.
- **`product`** is a species name that must exist in the gas model. The mass consumed by the species is added to this species.
- Species omitted from the dictionary get an efficiency of zero, so they do not recombine.

### Grid

| Key | Type / values | Default | Description |
| --- | --- | --- | --- |
| `CLUSTERING` | `false`, int, or float | `false` | A number greater than 1 is the Roberts clustering parameter, clustering cells towards the wall; values close to 1 cluster hardest. |

There are only two accepted values.

- **`false`**, or the key left out altogether, builds a uniform grid; fixed $\Delta x$.
- **A number greater than 1** clusters cells towards the wall. Values close to 1 cluster hardest.

### Time stepping and termination

| Key | Type / values | Default | Description |
| --- | --- | --- | --- |
| `T_END` | float, or dict `{steady = <residual>}` | `{steady = 1e-1}` | A float runs time-accurately and stops at that simulation time. The dictionary form runs in steady mode, stopping once the largest change in the solution has stayed below that residual for 100 iterations. |
| `MAX_ITERS` | int | `1e8` | Maximum number of iterations; the run stops and reports when it is reached. |
| `CFL_c` | float | `0.4` | Convective CFL number, $\Delta t_c = \mathrm{CFL}_c \, \Delta x / (\lvert u \rvert + a)$. |
| `CFL_d` | float | `0.4` | Diffusive CFL number, $\Delta t_d = \mathrm{CFL}_d \, \Delta x^2 / \max(\nu, \alpha, D_s)$. |
| `LTS` | bool | `false` | Local time stepping. Each cell advances on its own stable time step, which accelerates convergence but destroys time accuracy. |
| `HOT_START` | bool | `true` | Starts the run from an estimate of the post-shock state instead of a uniform freestream, which converges faster. |

`LTS` has two requirements, and the run stops if either is broken.

- It needs steady mode, so `T_END` must be given in its dictionary form; and
- it cannot be combined with transient output, that is, the dictionary form of `SAVE`.

### Output

| Key | Type / values | Default | Description |
| --- | --- | --- | --- |
| `SAVE` | `false`, a string, or a dict `{path = <string>, interval = <float>}` | `false` | Output control, described below. |
| `PLOT` | bool | `false` | Writes PNG plots after the run. Has no effect unless `SAVE` is set. |
| `VERBOSE` | bool | `true` | Prints progress, the iteration and residual trace every 1000 iterations, and the final heat flux and standoff. |

`SAVE` behaves in three ways.

- **`false`**, nothing is written.
- **A string**, the profile is written to that path as a CSV, and the stagnation-point summary.
- **A dictionary**, as above, plus a transient trace of the same stagnation-point quantities, written to `transient_<path>` at the given `interval` of simulation time. In this mode the `stagpoint_` file is *not* written, which matters when the run is driven by NCB (see [Output layout](#output-layout)).

The profile CSV columns are $x$, $u$, $\rho$, $T_{tr}$, $p$, $e_{tr}$, $e_{ve}$, $\mu$, $k_{tr}$, $a$, $\beta$, $T_{ve}$, $k_{ve}$, followed by one column per species. On a one-temperature gas model the
three vibro-electronic columns are omitted and the header reads $e$, $T$ and $k$.

With `PLOT` enabled, the plots written are `<name>_prim.png` for the primitives, `<name>_massf.png` if chemistry is on, and `<name>_transient.png` if the transient mode is in use, where `<name>` is `SAVE` with its extension stripped.

### A complete NESL job file

```toml
# Freestream.
T_INF = 300.0
P_INF = 100.0
U_INF = 5000.0
MASS_FRAC = {N2 = 0.767, O2 = 0.233}

# Geometry.
RADIUS = 0.05
DOMAIN = 0.005
AXISYMMETRIC = true

# Models.
GAS_MODEL = "air-5sp.gas"
CHEM_MODEL = "air-5sp.chem"
EXCHANGE_MODEL = "air-5sp.kin"

# Grid.
CELLS = 100
CLUSTERING = 1.05

# Wall.
T_WALL = 300.0
VISCOSITY = true
CATALYTIC = "equilibrium"
LEWIS_NUM = 1.2

# Time stepping.
T_END = {steady = 1e-2}
MAX_ITERS = 1e7
CFL_c = 0.4
CFL_d = 0.1
LTS = true

# Output.
SAVE = "result.csv"
PLOT = true
VERBOSE = true
```

Invoked as,

```
nesl --job=template.toml
```

---

## NCB bounds file settings

The bounds file defines the sweep. NCB takes the Cartesian product, or the element-wise zip, of the freestream arrays, generates one NESL job per combination, runs them in parallel, and collects the results.

**NCB is not configured to work with transient runs.** It only handles the steady stagnation-point output, so the template must use the plain string form of `SAVE`. See [Output layout](#output-layout).

### Keys

| Key | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `T_BOUND` | dict, string, or number | Yes | — | Freestream temperature values, K. |
| `U_BOUND` | dict, string, or number | Yes | — | Freestream velocity values, m/s. |
| `P_BOUND` | dict, string, or number | One of the two | — | Freestream pressure values, Pa. |
| `RHO_BOUND` | dict, string, or number | One of the two | — | Freestream density values, kg/m$^3$. |
| `PERMUTATIONS` | bool | No | `true` | `true` sweeps the full Cartesian product of the three arrays. `false` zips them element-wise, so entry $i$ of each array forms case $i$. |
| `DOMAIN_SCALE` | float | No | `1.5` | Safety factor on the estimated shock standoff used to set each case's `DOMAIN`. Ignored when `FIXED_DOMAIN` is set. |
| `FIXED_DOMAIN` | bool | No | `false` | `true` holds the domain fixed across the sweep: NCB estimates no standoff and leaves the template's `DOMAIN` untouched. |

One of `P_BOUND` or `RHO_BOUND` must be given. If `P_BOUND` is present it is used and the jobs are written with `P_INF`; otherwise `RHO_BOUND` is used and the jobs are written with `RHO_INF`.

Under `PERMUTATIONS = false` the arrays should be the same length; any extra entries are ignored.

### Bound specification

Each of `T_BOUND`, `U_BOUND`, `P_BOUND` and `RHO_BOUND` accepts three forms.

**A single number,** a scalar that is held fixed across the sweep.

```toml
T_BOUND = 300.0
```

**A range dictionary,** takes `lower`, `upper`, `cells` and `bound_type`.

```toml
U_BOUND = {lower = 4000.0, upper = 5000.0, cells = 5, bound_type = 'linear'}
```

| `bound_type` | Spacing |
| --- | --- |
| `'linear'` | `cells` points evenly spaced from `lower` to `upper`. |
| `'log_lower'` | `cells` points logarithmically spaced, ascending from `lower` to `upper`, so points bunch near `lower`. |
| `'log_upper'` | `cells` points logarithmically spaced, descending from `upper` to `lower`, so points bunch near `upper`. |

**A path string,** a text file of values, one per line, loaded directly.

```toml
T_BOUND = "temperatures.txt"
```

### What NCB writes into each job

For each case NCB copies the template and overrides:

- **`T_INF`**, `U_INF`, and either `P_INF` or `RHO_INF` from the sweep;
- **`DOMAIN`**, computed per case as `DOMAIN_SCALE` times the inviscid frozen normal-shock standoff estimate for that condition. Under `FIXED_DOMAIN = true` this key is not written at all and the template's own `DOMAIN` carries through to every case; and
- **`GAS_MODEL`**, plus `CHEM_MODEL` and `EXCHANGE_MODEL` where present, each rewritten to `../../<file name>`.

That last rewrite means the model files must sit in the directory NCB is launched from, since each case runs two levels down in `runs/caseNNN/`.

Everything else in the template, grid, wall, catalysis, time stepping, output, is passed through unchanged.

### Output layout

```
runs/
    case00/
        job.toml
        ... nesl output ...
    case01/
    results.csv
```

NCB refuses to start if `runs/` already exists.

`results.csv` collects one row per case: case number, `T_inf`, `p_inf` or `rho_inf`, `u_inf`, standoff, convective heat flux, diffusive heat flux, stagnation pressure and shock radius of curvature.

The template's `SAVE` must be a plain string. With `false`, or with the dictionary form that enables the transient trace, every row of
`results.csv` comes out as `NaN`.

### A complete NCB bounds file

```toml
T_BOUND = {lower = 200.0, upper = 300.0, cells = 3, bound_type = 'linear'}

RHO_BOUND = {lower = 1e-4, upper = 1e-2, cells = 4, bound_type = 'log_lower'}

U_BOUND = {lower = 4000.0, upper = 8000.0, cells = 5, bound_type = 'log_upper'}

PERMUTATIONS = true

DOMAIN_SCALE = 1.5
```

Invoked as,

```
nesl_condition_builder --job=template.toml --bounds=bounds.toml --max_cpus=8
```

---

## Improving stability

Settings worth adjusting first when a run will not converge.

- Lower `CFL_c` and `CFL_d`, e.g. to 0.1-0.2.
- Keep `SENSOR` and `HOT_START` on.
- Cluster the grid towards the wall with `CLUSTERING`, close to 1, e.g. 1.01-1.05.
- Increase `CELLS` to better resolve the shock and boundary layer.
- Check `DOMAIN` is long enough to contain the shock.
