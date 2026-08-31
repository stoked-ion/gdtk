# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GDTk (Gas Dynamics Toolkit) is a collection of compressible/reacting gas-dynamics and CFD tools from the University of Queensland's Centre for Hypersonics. The simulation codes are written in **D**, configured by user-supplied **Lua** input scripts, and exposed to users through loadable **Python**/**Ruby** libraries. The bias is toward chemically-reacting, high-temperature, hypersonic flows (shock tunnels, expansion tubes).

## Repository layout (`src/`)

Two generations of the main flow solver coexist and share the same underlying physics libraries:

- **`eilmer/`** — Eilmer 4, the established finite-volume compressible flow solver. Builds `e4shared` (shared-memory), `e4mpi` (distributed), and complex/steady-state variants. Input via Lua + `e4shared --prep/--run/--post`.
- **`lmr/`** — Eilmer 5 ("lmr"/lorikeet), a reorganization of Eilmer with a single `lmr` executable using a **subcommand** dispatch (see `src/lmr/main.d`; subcommands live in `src/lmr/commands/`). This is where most new solver development happens.

Shared libraries used by both solvers and the standalone tools:

- **`gas/`** — gas models (ideal, CEA, equilibrium, thermally-perfect, multi-temperature). `prep-gas` compiles `.lua` gas inputs into gas-model files.
- **`kinetics/`** — finite-rate chemistry and thermal energy-exchange. `prep-chem`/`prep-reactions` and `prep-kinetics`/`prep-energy-exchange`.
- **`geom/`** — geometry primitives, paths, surfaces, structured/unstructured grids.
- **`nm/`** — numerical methods (linear algebra, root finding, integration). Defines the `number` abstraction (see below).
- **`gasdyn/`** — analytic gas-dynamic relations (normal/oblique/conical shocks, isentropic flow).
- **`ntypes/`** — `complex.d`, the `Complex!double` type used for complex-step differentiation.
- **`util/`**, **`extern/`** — utilities and vendored dependencies (Lua 5.4.3, OpenMPI bindings, eqc, gzip).

Standalone tools each have their own directory: **`l1d/`** (shock-tunnel/expansion-tube end-to-end sim), **`nenzf1d/`**, **`pitot3/`**, **`onedval/`**, **`puffin`/`slf`** (space-marching), **`chicken/`** (GPU solver).

User-facing loadable libraries live in **`src/lib/gdtk/`** (Python package + Lua/Ruby modules): `gas.py`, `ideal_gas_flow.py`, `reflected_shock_tunnel.py`, `lmr.py`, etc.

## Building and installing

Each `src/` subdirectory has its own `makefile`. Build a tool by running `make install` from its directory; this also builds its library dependencies. The default install tree is `$HOME/gdtkinst` (override with `INSTALL_DIR=...`). The install tree is **separate** from the repository tree.

```bash
cd src/lmr && make install        # build & install Eilmer 5 (lmr)
cd src/eilmer && make install      # build & install Eilmer 4
make PLATFORM=macosx install       # macOS needs this hint
```

After installing, the user's environment must define (see README.md):
```bash
export DGD=$HOME/gdtkinst
export DGD_REPO=$HOME/gdtk
export PATH=$PATH:$DGD/bin
export DGD_LUA_PATH=$DGD/lib/?.lua
export DGD_LUA_CPATH=$DGD/lib/?.so
```

Compiler: **`ldc2`** (LLVM D compiler) by default; `DMD=dmd` is also accepted. A C compiler, gfortran, and (for MPI builds) OpenMPI are required.

### Key make variables

These select compile-time variants — changing them requires a rebuild (`make clean` first):

- `FLAVOUR` = `debug` (default; runtime checks, detailed errors) | `fast` (optimized production) | `profile`.
- `WITH_MPI=1` — build the distributed-memory executables (`e4mpi`, `lmr-mpi-run`).
- `WITH_COMPLEX_NUMBERS=1` / lmr's `lmrZ*` targets — build the complex-number version (see below).
- `MULTI_SPECIES_GAS`, `MULTI_T_GAS`, `MHD`, `TURBULENCE`, `NK` — toggle physics modules in/out for smaller/faster builds. Default on.
- `WITH_NK=1` (eilmer) — Newton-Krylov steady-state solver.

## The `number` type abstraction (important)

Physics/numerics code is written generically over the alias **`number`** (defined in `src/nm/number.d`):

```d
version(complex_numbers) { alias number = Complex!double; }
else                     { alias number = double; }
```

The whole solver is compiled **twice**: a real-valued build for normal simulation, and a complex-valued build (`e4z*`, `lmrZ*`) used for **complex-step differentiation** to construct numerical Jacobians and design sensitivities (adjoint/shape optimization). When editing solver code, keep it valid for both: use `number` rather than `double` for flow quantities, and prefer functions that work under both versions. Many bugs surface only in the complex build, so unit tests run in both modes.

## Testing

**D unit tests** are embedded `unittest` blocks compiled into a test runner (`src/util/test_runner.d`). A library directory with a `test`/`test-real`/`test-complex` make target (e.g. `src/nm/`) runs them:

```bash
cd src/nm && make test          # runs both real and complex unit tests
cd src/nm && make test-real     # real-valued only
cd src/nm && make test-complex  # complex-valued only
```

`make demo` in library dirs builds small standalone demo programs.

**Integration/regression tests** live alongside the examples and are driven by Ruby/Tcl/Python scripts:

```bash
cd examples/eilmer.test && ./eilmer-test.rb   # full suite (~1.5 h); needs ruby, python-sympy
```
Many example subdirectories contain `*-test.rb` / `test_*.rb` scripts that run a case and compare against golden results. Run them from a copy of the examples tree, not in-place (they generate output).

## Typical lmr (Eilmer 5) simulation workflow

```bash
lmr prep-gas -i gas.lua -o gas.gas        # compile gas model
lmr prep-reactions -g gas.gas -i chem.lua -o chem.chem
lmr prep-grid                              # default job file: job.lua
lmr prep-sim
lmr run                                    # shared memory; mpirun -np N lmr-mpi-run for MPI
lmr snapshot2vtk --add-vars="mach"         # post-process to VTK for ParaView
```
`lmr help -a` lists all subcommands. The Eilmer 4 equivalent is `e4shared --prep/--run/--post --job=<name>` after a standalone `prep-gas`.

## Conventions

- **Indentation:** 4 spaces for D, **3 spaces for Lua**; spaces not tabs (tab stops assumed every 8 columns). An `.editorconfig` is in `doc/editorconfig`.
- **Commit messages** are prefixed with the affected area, e.g. `lmr: ...`, `gas: ...`, `examples/gas: ...`. A `doc/prepare-commit-msg` git hook can auto-suggest the prefix.
- Run `make clean` before committing so build artifacts (`*.o`, executables, the vendored Lua build under `extern/lua-5.4.3/`) don't get added. Do not commit regenerable binaries.
- History is kept close to linear; prefer small, single-issue commits and pull immediately before pushing.

## MHD capability in lmr (Eilmer 5)

There are **three separate "MHD" pathways** in `lmr`, with very different maturity. This matters because the current work (`gdtk-mhd` branch) does MHD via user-defined source terms, not the built-in MHD flag.

### 1. UDF source terms — low-Rm / imposed-field MHD (recommended, no MHD flag needed)

`getUDFSourceTermsForCell` (`src/lmr/user_defined_source_terms.d:25`) reads the Lua table returned by the user's `sourceTerms(t, cell)` function and adds it to the cell source vector via `add_udf_source_vector` → `Q.add(Qudf)` (`src/lmr/fluidfvcell.d:1296`). The generic fields `momentum_x/y/z`, `total_energy`, `mass`, `species`, `energies` are applied **regardless of compile flags** (`user_defined_source_terms.d:57-61`). This is the supported way to impose a Lorentz force **J×B** (momentum source) and Joule heating (energy source).

What the UDF cell table exposes (`pushFluidCellToTable` → `pushFlowStateToTable`, `src/lmr/luawrap/luaflowstate.d:350`): position, `vol`, `p`, `T`, `rho`, `vel.x/y/z`, `a`, `mu`, `k`, `massf`, `T_modes`, and **`sigma`** (electrical conductivity, populated if a `conductivity_model` is set). The magnetic field `B`, `psi`, `divB` are exposed **only** under `version(MHD)` (`luaflowstate.d:384-388`); without the MHD build the user must supply **B** themselves in the UDF (the normal low-magnetic-Reynolds-number assumption).

### 2. Electric-field Poisson solver (`src/lmr/efield/`) — always compiled, not behind `version(MHD)`

Solves a Poisson equation for electric potential given a conductivity model, wired into the transient loop (`src/lmr/timemarching.d:444`). Enable via `config.solve_electric_field = true`, `config.electric_field_count = N`, `config.conductivity_model_name` (`test` | `constant` | `raizer` | `diffusion` | `none`, see `efieldconductivity.d:118`). Runnable example: `examples/lmr/2D/efield-solver/`. Use it to compute a self-consistent current distribution instead of prescribing **J**.

#### Electrode boundary conditions for the field solver (`efieldbc.d`)

The field solve needs a boundary condition at each electrode. Three exist, in increasing order of what they let the electrode's metal potential do:

- **`FixedField`** — prescribed potential (optionally a profile via `Ex_applied`/`Ex_quad`/`Ex_cube`). No electrode impedance at all.
- **`SheathField`** — prescribed metal potential `Velectrode`, with a physical sheath impedance in series (`sheath_model = "linear" | "child-langmuir"`; `K` is the Child–Langmuir coefficient, ~500 for argon). Supports segmented electrodes (`segment_pitch`/`segment_fill`/`segment_x0`) and an axial tilt of the metal potential (`Ex_applied`/`Ex_quad`/`Ex_cube`, the diagonal-mode angle).

  **`segment_pitch` is geometric segmentation only.** It cuts the metal into strips but holds every strip at the *same* `Velectrode`, so φ is still pinned to one value wherever metal exists and **no axial electrical isolation is gained**. Measured: refining the pitch 25 → 12.5 → 6.25 mm at β ≈ 2.9 leaves the electrode alignment `E*_x/(β E*_y)` flat at ~0.50 and F_x at 149.1 / 144.6 / 147.2 N/m. If you want segments that can float to different potentials, you need per-pair supplies — `CircuitElectrode` with a node per pair.

  **A tilt must be ramped on, not imposed.** At condition 6 the optimum tilt is 1838 V across a duct driven at 400 V, and applying any tilt in one step destroys the solve — measured, the run dies within ten steps even when restarted from a fully converged field and run first order. `Ex_ramp_steps` smoothsteps the tilt on over that many Newton steps starting at `Ex_ramp_start` (both 0 by default, which reduces exactly to the previous behaviour). This is the same remedy a deferred UDF source needs and for the same reason: the steady solver passes `t = -1`, so the ramp is on `SimState.step`.

  **`Ex_applied` is the lever that does move alignment.** `V(x) = Velectrode + Ex_applied·dx`, so the imposed axial field is `−Ex_applied`; the classical optimum is `tan θ = β`, i.e. `E*_x = β E*_y`, at which `J_x = 0` and the `(1+β²)` cancels. Verified at β = 2.86: alignment 0.52 → 0.81 → 1.76 as the tilt goes 0 → −514 → −1070 V/m, and `I_x/I_y` crosses zero (−0.083 → −0.023 → +0.041) exactly where alignment reaches 1. **The thrust gain is only 2.6%**, because the sheath trades `E*_y` for `E*_x` almost 1:1 (374 → 133 V/m while `E*_x` goes 556 → 667), leaving `(E*_y + β E*_x)/(1+β²)` nearly unchanged. Nulling `J_x` is real physics and worth little at β ≈ 3. **At β ≈ 15 it is worth much more**: a 92 V tilt (`Ex_applied = -918`, 5% of the classical optimum) raises the alignment from 0.064 to 0.151 and `F_x` from 339.6 to 379.2 N/m, **+11.7%**, with `Te_max` *below* the untilted case. Larger tilts do not converge — 184 V drifts and grows a 39 kK spot at the upstream anode corner, and 276 V crashes at the end of its ramp. Reaching the classical optimum at this β is out of reach of the present numerics. **This is the right BC for Faraday and diagonal** — independent electrode pairs, and a resistor-ladder tilt, are both correctly modelled by a prescribed metal potential. Do not migrate those cases.
- **`CircuitElectrode`** (`efieldcircuit.d`) — the metal potential is an **unknown**, namely the potential of node `node` in `config.external_circuit`. Needed whenever electrodes are wired **to each other** (a Hall short, or segments sharing a ballast network): prescribing both terminals' potentials supplies no equation limiting the current between them, which is what made every earlier Hall attempt diverge. Two electrode groups naming the same node id are thereby shorted together.

`config.external_circuit` is a Lua table of `nodes` (each with a `nominal_voltage`, used to seed the first sheath linearization) and `resistors` (`{a=i, b=j, R=...}` node-to-node, or `{a=i, R=..., V_supply=...}` for a leg to a fixed supply). Resistances are in **Ω·m of depth**, since the 2-D solve carries currents per metre of depth. It may also carry `sources` (`{a=i, I_supply=...}`), ideal **current sources** injecting a fixed current (A per metre of depth) into a node — the right drive for a Hall connection, whose terminal sheaths otherwise absorb most of an applied voltage. A current source stamps only the right-hand side, so it adds no conductance and **cannot anchor the potential level**: a current-driven circuit still needs one voltage reference. The working arrangement is a current source on the driven node and a **stiff supply leg on a different node as a grounded return** (`{a=j, R=2e-3, V_supply=0}`) — how a real current-regulated supply is wired. Two opposed current sources (`+I`, `−I`) with a *high-resistance* reference leg does **not** work and fails slowly rather than loudly: the whole circuit drifts (measured: to −4.7e8 V, relaxing 5% per field solve), because the leg carries no current at convergence and so exerts almost no restoring force on the level. A stiff leg on the *same* node as a current source is equally wrong — it just shunts the source. A node needs either a resistive path to a supply leg or at least one electrode face (its sheath conductance ties it to the plasma, so an isolated electrode pair may float); the circuit as a whole needs at least one supply leg, or it is gauge-singular.

Implementation: the electrode unknowns border the existing 5-band matrix, and the augmented system is solved by a Woodbury/Schur complement (`schurSolve`) — `K+1` solves of the *unmodified* banded operator, so the existing GMRES/ILU path is untouched. Measured cost ≈1.23×/step for `K=2`. With no circuit declared, the code takes exactly the pre-existing path. Correctness anchor: a Faraday case re-expressed as `CircuitElectrode` with `R → 0` reproduces the `SheathField` result to 0.05% in F_x and total current — kept as Tier 0 of the project regression suite.

#### A non-uniform applied magnetic field (`config.applied_B_ramp`)

`config.applied_Bz` is the peak field. By default it applies everywhere, which is fine at low Hall parameter but not at high: a uniform field running to an insulating outflow forces the whole axial Hall current to turn around inside the last cells, giving a current concentration that is real physics but an artefact of truncating the duct rather than a property of the device. Set `applied_B_ramp` (and `applied_B_x0`/`applied_B_x1`) for the usual fringe-field window

```
B(x) = applied_Bz * 0.5*(tanh((x - x0)/L) + tanh((x1 - x)/L))
```

— roughly `applied_Bz` between the magnet edges, half of it at each edge, decaying over `L`. `applied_B_ramp = 0` (the default) returns the constant everywhere and takes exactly the pre-existing code path. Measured on C6_pow at β≈15: the peak `|J|`-to-bulk ratio falls monotonically with taper length (59 → 11 → 9.4 → 6.3 → 4.8 for L = 0, 4, 4, 8, 12 mm) and the maximum moves off the boundary onto the magnet edge.

**A UDF that forms J×B itself must read `cell.Bz_applied`, not a hard-coded constant** — otherwise its current disagrees with the solved potential everywhere outside the flat region. It equals `config.applied_Bz` when no taper is set.

**Check B/B0 at the electrodes before interpreting a taper run.** A magnet that leaves electrodes outside the field is not a small perturbation: with the first and last electrode pairs at B/B0 = 0.03–0.22, C6's F_x collapses by 93%, because an unmagnetised electrode shorts the plasma while generating no EMF.

With the magnet covering the electrodes and rolling off before the duct exit, two converged runs at β≈2.85 differ only in the field profile: F_x 179.5 → 137.8 N/m (−23%), I −23%, Δu −17%, P_J 476 → 570 kW/m (+20%). **The uniform-field idealisation overstates thrust and current by roughly a quarter and understates dissipation by a fifth.**

Switching a large source on mid-run needs three things beyond the profile itself: a step-based ramp in the UDF (`cell.step`, since the steady solver passes `t = -1`), a first-order low-CFL Newton-Krylov phase to absorb the transition, and `reset_reference_residuals = true` on the phase *after* the ramp — otherwise the auto-CFL stays frozen (see below). With all of these, C6 runs at β≈16 in second order.

#### Switching a source on part-way through a steady run

Three separate traps, all of which bite together:

- **The steady solver passes `t = -1`**, so a UDF source ramped on physical time evaluates to zero for the whole run. Ramp on **`cell.step`** instead (available in the source-terms table; it is the Newton step in a steady run, the time step in a transient one). Use a smoothstep — a linear ramp puts a kink in the residual at each end, which a Newton solver feels.
- **The auto-CFL is frozen for the rest of the run.** `ResidualBasedAutoCFL` returns the current CFL unchanged while the *relative* residual exceeds the growth threshold, and the reference residuals are taken over the first ~10 steps, before the source exists. Afterwards the relative residual sits at 1e4–1e6 permanently. Set **`reset_reference_residuals = true`** on a `NewtonKrylovPhase` to re-take them — on the phase that begins *after* the ramp finishes, not where it starts (ten steps into a 600-step ramp captures 1.7% of the source and the relative residual plateaus near 500). Then restore the growth threshold to the default ~0.99, since the relative residual starts from 1 again. Note this also changes what `stop_on_relative_residual` means.
- **Second-order reconstruction may not survive the transition.** The signature is `update_thermo_from_rhou` reporting negative internal energy. A first-order, working-CFL phase spanning the ramp and the relaxation after it is the fix; `extrema_clipping` does not help, and `thermo_interpolator = "pT"` removes the hard failure but not the underlying stiffness.

#### Hall discretisation and the insulator boundary condition (`LMR_HALL_SCHEME`, `LMR_INSULATOR_BC`)

Two environment switches select how the Hall (skew) part of the conductivity tensor is discretised. Both default to the historical behaviour, and every established result on this branch was produced with the defaults — a fresh converged `C6_pow` run under `central` with all of the below in place reproduces the stored golden F_x, F_y, Δu, I and η to 0.000%.

- **`LMR_HALL_SCHEME=central`** (default) — the original centred full-tensor flux, `σ_H (t·∇φ)`. Each of the two cells sharing a face reconstructs that tangential gradient from its own stencil, so it is two-valued: a discrete-curl defect that manufactures current wherever the stencil family changes.
- **`LMR_HALL_SCHEME=upwind`** — Path 2, the Parent et al. (2011) reformulation. Integration by parts turns the Hall term into an exactly conservative upwinded convective flux, `−S(a·n)φ_face` with `S(a·n) = σ_H(v_end) − σ_H(v_start)` between the face's two *vertices*: single-valued, and it telescopes to exactly zero around a closed cell. The implicit operator stays 5-point, so Path 1's Woodbury path is untouched. Pass it through MPI with `mpirun -x LMR_HALL_SCHEME`.

Under the split, two boundary treatments that were adequate for the scalar path stop being correct, because a face's Hall flux is no longer a local quantity:

1. **Insulating (`ZeroNormalGradient`) faces.** `J·n = 0` with a skew tensor is *not* `∂φ/∂n = 0`. Writing `J·n = m·(−∇φ + u×B)` with `m = σ_P n + σ_H t`, `t = (n_y, −n_x)`, and dividing by σ_P gives an oblique-derivative (mixed Robin) condition:

   ```
   ∂φ/∂n = −β·∂φ/∂t + (u×B)·n + β·(u×B)·t
   ```

   Transcribe the signs from the code's own `m`, not from a textbook frame — `(n, t, ẑ)` as defined here is left-handed. The `ZG*` one-sided families are derived for a *zero* wall slope, but their 4×4 system is linear, so a non-zero slope `g_n` simply adds `g_n·C` to the reconstructed gradient, `C = (ZG?_Gx, ZG?_Gy)/D` (`efieldderivatives.d`), with `C·n ≡ 1` exactly. Since `g_n` itself depends on `∂φ/∂t`, the loop closes in one substitution — `g_n = (s − β t·∇φ_h)/(1 + βγ)`, `γ = C·t` — so the whole condition becomes a rescaling of the cell's own stencil weights: fully implicit, in-band, no lagging, no bandwidth change. The old condition is not merely inaccurate, it is **inconsistent**: it does not converge under grid refinement (129% error at β = 15 on an insulating wall, and *growing* with refinement at an insulating end).

   **`LMR_INSULATOR_BC`** selects the model: `tensor` (default under upwind) is the full condition, right for a genuinely insulating surface sitting in the field; `emf` drops β, `∂φ/∂n = (u×B)·n`, which is the right model for an **open end** truncating a longer channel, since it lets the Hall current continue through the boundary rather than turning it back into the domain; `legacy` restores `∂φ/∂n = 0` with no wall source. This is a *modelling* choice, not a discretisation detail — it changes the end-effect current loops.

2. **Electrode (`SheathField`/`CircuitElectrode`) faces.** Zeroing a face's stencil removes its Pedersen flux and its source, which under `central` removed the whole tensor flux and so did impose `J·n = 0`. Under the split it does not: the face's Hall flux is in the contour sum, which cannot be gated without breaking the telescoping. The face must instead subtract its own `σ_H (t·∇φ)`, i.e. take the stencil factor `−σ_H t`. Use the **vertex-averaged** σ_H that the convection term uses — a cold electrode's *face* conductivity has collapsed to ~0 while its vertices carry the hot near-wall cell value.

**Status.** With both corrections the scheme is exact to round-off on a `J = 0` exact solution at β = 15, for all four `ZG*` families and both wall orientations. At β≈16 it runs in second order and is converged for engineering purposes: it sits on a ~5e-3 relative residual floor, but that is a Hall-term limit cycle confined to a near-wall band in the current-turning region — the core is converged to 1e-7, no integrated quantity moves by more than 0.13%, and the identical case with `electric_field_hall_effect = false` converges to 7e-11. If it ever needs removing, the target is the near-wall Hall stencil, not the CFL or the field–flow coupling. At β ≈ 1.4 in the solver it agrees with `central` to 0.09% in F_x and total current, and the boundary condition is directly confirmed there: median |J_x| in the end cells falls from 1.50e5 (against a bulk 1.62e5 — i.e. the old condition lets the full Hall current flow straight out of the domain) to 5.98e3, with the bulk unchanged. It still fails at β ≈ 15 on a **cold, σ-collapsing electrode wall**; that residual error is the first-order donor-cell upwinding — Parent's minmod anti-diffusion (his term 4) is not implemented anywhere in the tree, and is the next piece of work. Standalone verification harnesses: `tools/hall-stencil/zngbc.py` (recovers and checks the `ZG*` Maxima conventions, derives the `_Gx/_Gy` coefficients) and `tools/hall-stencil/channel.py` (assembles the whole scheme against an exact solution; this is where to reproduce a suspected boundary defect cheaply, rather than in a 10-minute solver run).

### 3. Built-in single-fluid MHD (`version(MHD)`, default on via `MHD ?= 1`) — present but rough in lmr

The Bond/Wheatley single-fluid model, ported from Eilmer 4 (README: "a work in progress"). It is wired through conserved quantities (adds `xB, yB, zB, psi, divB`, and forces z-momentum on in 2D — `conservedquantities.d:133-184`), config keys (`config.MHD`, `MHD_static_field`, `MHD_resistive`, `divergence_cleaning`, `c_h`, `divB_damping_length` — `lua-modules/globalconfig.lua:30`), HLLE flux + Dedner divergence cleaning (`fluxcalc.d:142`), and explicit-update divergence damping (`simcore_gasdynamic_step.d:1122`). **Caveats:**

- **Transient explicit only.** The implicit / Newton-Krylov steady path has unfinished `// [TODO] PJ 2021-05-15 MHD bits` (`simcore_gasdynamic_step.d:2442`, `:2854`).
- **No `lmr` examples.** All MHD examples ship under Eilmer 4 (`examples/eilmer/2D/mhd-blunt-nose`, `MHDShockTube`, `mhd-kelvin-helmholtz`); none under `examples/lmr/`, so the lmr MHD path is essentially untested by the suite.
- **Known bug:** `fluxcalc.d:149` writes the z-field divergence-cleaning flux into `F[cqi.xB]` instead of `F[cqi.zB]` (double-hits `xB`, never sets `zB`).
- Setting `config.MHD=true` without compiling `MHD=1` throws at runtime: *"MHD capability has not been enabled"* (`globalconfig.d:2071`).

#### Which electrode connection to use

Established by measurement on this branch, not assumed: in an argon duct at β ≈ 1–2.6, **only the Faraday connection accelerates** — Hall and diagonal are generators at every drive level and every field strength tested, and diagonal gets worse with drive. The Hall connection short-circuits each electrode pair transversely, so the sheath is its only transverse impedance and the transverse bracket in `J_y = σ/(1+β²)[E_y + βE_x − B u_x]` is pinned near zero regardless of drive; a current-driven control over a 12.5× current range and both polarities moves `F_x` by less than 50%. The Hall parameter also collapses under its own drive current (β 1.86 → 0.65 as Joule heat raises the collision frequency), so `βE_x` saturates well below `B u_x`. Hall/diagonal additionally need an electrode pitch `ℓ ≲ H/β` to hold an axial field at all — and note that `segment_pitch` alone does not deliver that, see above. A standalone Hall drive breaks even only when `μ_e·E_x > u` (the axial electron drift outruns the gas); that condition is **independent of B**, and measurement puts this device at 0.23–0.62 of it, saturating, because the drive current's own Joule heating raises the collision frequency. Reach for `CircuitElectrode` when electrodes are wired to each other; use `SheathField` for Faraday and diagonal.

**Guidance:** prefer the UDF approach (pathway 1); keep `config.MHD` off and supply B in the UDF; optionally enable the efield solver (pathway 2) for a computed current. Treat the built-in MHD (pathway 3) as transient-only and unvalidated in lmr.

## Documentation

User guides (PDF/AsciiDoc) are under `doc/` (`lmr-reference-manual.adoc`, `geometry-reference-manual.adoc`, `nm-reference-manual.adoc`, etc.) and at <http://gdtk.uqcloud.net>. `doc/lmr-cheatsheet/` has a quick command reference. `doc/developer-notes.md` covers the git/dev workflow in detail.
