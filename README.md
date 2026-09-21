# Viscoacoustic modelling, Tapenade FWI, and LSM

This checkout builds the MPI application `bin/SMIwiz` — an integrated toolbox
for 2D/3D seismic forward modelling, RTM imaging, linearized and nonlinear
full waveform inversion, and source time-function estimation. The forward
propagator supports an isotropic acoustic step and an optional three-standard-
linear-solid (SLS) viscoacoustic step. Algorithmic differentiation (AD) via
Tapenade 3.16 generates the tangent-linear and adjoint routines required by
FWI and least-squares migration (LSM/ LSRTM).

## Runtime modes

The executable supports these `mode` values:

- `mode=0`: forward modelling. Computes synthetic shot records from the
  supplied velocity, density, and (optionally) attenuation model.
- `mode=1`: nonlinear FWI. Iteratively minimizes the data misfit with
  NLCG, l-BFGS, or Newton-CG using the Tapenade-generated gradient.
- `mode=2`: RTM. Produces a reverse-time migration image through the same
  impedance-gradient kernel as the FWI adjoint (`do_fwi()` with mode=2).
- `mode=3`: matrix-free linearized waveform inversion (LSM / LSRTM). Solves
  the normal equations with CG, or the L1-regularized problem with FISTA,
  using the generated TLM and adjoint operators.
- `mode=4`: standalone FWI gradient evaluation. One objective and gradient
  computation without optimization iterations — used for gradient checking
  and sensitivity analysis.
- `mode=5`: source time-function inversion (Pratt, 1999). Estimates the
  source wavelet from weighted observed data in the frequency domain via FFT.

Other mode values are rejected at startup with an error message.

## Data objective

The scalar misfit minimized in modes 1 and 4 is:

```text
J = 0.5 * sum_{receiver,time} [wdat * (dcal - dobs)]^2
```

`modelling()` is the nonlinear map from the physical model to `dcal`. It does
not receive observations, data weights, or a scalar objective. Tapenade
differentiates `dcal` with respect to the physical `vp`, `rho`, and `qinv`
inputs. `fg_fwi()` accumulates `J` in double precision, seeds the reverse
sweep with `dcalb = wdat^2*(dcal-dobs)`, and maps the returned physical-model
derivatives to the selected logarithmic `vp-rho` or `vp-ip` parameters.
Because the objective is outside the differentiated call graph, this
normalization changes the external reverse seed but not the Tapenade head or
the generated Jacobian-transpose implementation.

## Requirements

- An MPI implementation with C development support. `mpicc` must compile the
  sources and `mpirun` must be available to launch multi-rank runs.
- FFTW3 development headers and library. The selected prefix must contain
  `include/fftw3.h` and `lib/libfftw3.so` (or the platform-equivalent library).
- Tapenade 3.16 with a Java runtime supported by that release. The selected
  `TAPENADE_HOME` must contain both `bin/tapenade` and the `ADFirstAidKit/`
  directory used to compile `adStack.c` and `adBinomial.c`.
- Python 3 with `numpy` and `matplotlib` for the run-script validation and
  convergence plotting.

On Debian/Ubuntu, the system dependencies can be installed with:

```sh
sudo apt install openmpi-bin libopenmpi-dev libfftw3-dev default-jre
```

Install Tapenade separately, then verify the toolchain before building:

```sh
mpicc --version
mpirun --version
test -f /usr/include/fftw3.h
test -x /path/to/tapenade_3.16/bin/tapenade
test -d /path/to/tapenade_3.16/ADFirstAidKit
```

The Makefile defaults to `fftw3=/usr` and
`TAPENADE_HOME=/home/pyang/Install/tapenade_3.16`.

## Build

From the `viscoacoustic` directory:

```sh
make -C src
```

The default target removes `src/tapenade_out/`, regenerates the Tapenade
reverse and tangent sources, repairs the generated C code, then rebuilds and
links `bin/SMIwiz`. To only remove build products and generated Tapenade
files, run `make -C src clean`.

For a different installation layout:

```sh
make -C src \
  fftw3=/path/to/fftw3 \
  TAPENADE_HOME=/path/to/tapenade_3.16
```

### Generated AD sources

The build generates and repairs these Tapenade reverse sources before linking:

```text
src/tapenade_out/modelling_b.c
src/tapenade_out/extend_model_b.c
src/tapenade_out/fdtd_b.c
src/tapenade_out/inject_extract_b.c
```

And these tangent-linear sources:

```text
src/tapenade_out/modelling_d.c
src/tapenade_out/extend_model_d.c
src/tapenade_out/fdtd_d.c
src/tapenade_out/inject_extract_d.c
```

Both sets are generated from the same Tapenade head:

```text
modelling(dcal)/(vp rho qinv)
```

Every model/state argument has a matching directional-variation argument
(in the tangent sources) or adjoint argument (in the reverse sources). The
generated `dcalb` argument is an explicit input seed — FWI supplies the
weighted residual derivative while LSM applies J^T W^T W directly.

The Makefile strips Tapenade-ignored blocks from the primal source and passes
only the filtered copies to Tapenade. This keeps MPI timing instrumentation
and profiling (e.g. `MPI_Wtime()`, `time_info.txt`) available in the
application build without letting Tapenade parse or differentiate MPI side
effects. The filtering is controlled by `/* TAPENADE-IGNORE-BEGIN */` and
`/* TAPENADE-IGNORE-END */` markers in `src/modelling.c`.

To force regeneration after changing modelling kernels:

```sh
make -B -C src adjoint-generate tlm-generate
make -C src SMIwiz fftw3=/usr
```

### Post-generation repairs

Tapenade emits pointer-array assignments like `***pb = 0.0;` that are not
valid C for multi-dimensional arrays allocated as flat blocks.
`src/fix_tapenade_c.sh` replaces these with `memset()` calls and fixes the
`adBinomial_next()` API for the reverse checkpointing. It has two modes:

- `reverse`: repairs the reverse sources (`modelling_b.c`, `extend_model_b.c`)
- `tangent`: repairs the tangent sources (`modelling_d.c`, `extend_model_d.c`)

Both modes add `#include <string.h>` and replace every
`***array = 0.0;` with an appropriate `memset()` call.

## Source file overview

| File | Purpose |
|------|---------|
| `main.c` | MPI init, parameter parsing, per-mode dispatch |
| `modelling.c` | Tapenade-facing acoustic forward loop with binomial checkpointing |
| `fdtd.c` | FDTD kernels: velocity and pressure updates, 4th/8th order, CPML |
| `fdtd_split.c` | Loop-split FDTD — separates interior (no PML branches) from boundary faces |
| `inject_extract.c` | Source injection, receiver wavefield extraction, adjoint source injection |
| `extend_model.c` | Pad model domain with absorbing layers, compute kappa and buoyancy |
| `cpml.c` | CPML absorbing boundary coefficients |
| `do_modelling.c` | Mode-0 driver: CFL check, init, call modelling(), write data |
| `fg_fwi.c` | FWI objective/gradient: set model, primal run, adjoint seed, reverse sweep, chain-rule to log parameters |
| `do_fwi.c` | Mode-1 and mode-4 driver: optimizer loop with NLCG / l-BFGS / Newton-CG |
| `do_rtm.c` | Mode-2 driver: thin wrapper that calls `do_fwi()` to form the impedance-gradient image |
| `do_lsm.c` | Mode-3 driver: CG normal-equations solve and FISTA L1-regularized solve |
| `do_invert_source.c` | Mode-5 driver: source wavelet estimation in frequency domain |
| `gradient_check.c` | `checkgrad=1` second-order centered finite-difference validation |
| `opt.c` | l-BFGS two-loop recursion, line search (Wolfe conditions), NLCG, Newton-CG |
| `visco_lsq.c` | SLS coefficient fitting for constant-Q approximation |
| `precondition.c` | Implicit-reparameterization preconditioner (triangle smoothing + depth weighting) |
| `smoothing.c` | Box-triangle smoothing (Claerbout) in 2D/3D |
| `kaiser_windowed_sinc.c` | Kaiser-windowed sinc interpolation for source/receiver positioning |
| `acq_init_free.c` | Acquisition geometry from text file, interpolate to grid |
| `setup_data_mask.c` | Data weights and muting masks |
| `read_write_data.c` | Binary and SU format I/O for data and model vectors |
| `check_cfl.c` | CFL stability check |
| `fix_tapenade_c.sh` | Post-processing of Tapenade-generated C code |

## Input parameters

Parameters are passed on the command line as `key=value` pairs, typically
sourced from a parameter file with `$(cat inputpar.txt)`.

### Grid and time

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n1` | (required) | Number of grid points in depth (z / 1st axis) |
| `n2` | (required) | Number of grid points in x (2nd axis) |
| `n3` | 1 (2D) | Number of grid points in y (3rd axis) |
| `d1` | (required) | Grid spacing in depth |
| `d2` | (required) | Grid spacing in x |
| `d3` | (required) | Grid spacing in y |
| `nt` | (required) | Number of time steps |
| `dt` | (required) | Temporal sampling interval (s) |
| `nb` | 20 | Number of CPML absorbing layers |
| `nsnap` | 20 | Binomial checkpoint budget (1–98 when `nt>1`) |
| `order` | 4 | FD order: 4 or 8 |
| `freesurf` | 0 | 1 = stress-free surface condition; 0 = no free surface |
| `fm` | (depends) | Dominant frequency for CPML parameterisation |
| `dr` | 1 | Grid decimation for data output (unused in current kernels) |

### Input files

| Parameter | Required | Description |
|-----------|----------|-------------|
| `vpfile` | yes | Binary file of P-wave velocity (float, n1×n2×n3) |
| `rhofile` | yes | Binary file of density (float, n1×n2×n3) |
| `stffile` | yes* | Binary file of source time function (float, nt samples) |
| `qinvfile` | when `nrelax>0` | Binary file of attenuation 1/Q (float, n1×n2×n3) |
| `acquifile` | unless `suopt=1` | Text file listing source and receiver positions |
| `bathyfile` | optional | Binary file of bathymetry (float, n2×n3), water-bottom depth in grid indices |

`stffile` is not required in mode 5 (source inversion). When `eachopt=1`,
wavelets are read from `stffile_NNNN` where NNNN is the shot index.

### Viscoacoustic modelling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nrelax` | 0 | Number of SLS mechanisms: 0 (acoustic) or 3 (viscoacoustic) |
| `Q0` | (depends) | Target quality factor for constant-Q SLS fit |
| `sls_fmin` | 1 | Minimum frequency for SLS fit (Hz) |
| `sls_fmax` | (depends) | Maximum frequency for SLS fit (Hz) |

When `nrelax=3`, the three memory-variable SLS scheme is activated. The
coefficients are computed at runtime by `visco_sls_fit()` and stored in
`sim->sls_wl`, `sim->sls_yl`, `sim->sls_exp`, `sim->sls_1_exp`.
The FDTD pressure update includes the relaxation term:

```text
p -= dt * kappa * (div(v) - relax_term)
```

where `relax_term` combines the trapezoidal-rule SLS memory variables scaled
by `qinvmod`.

### Acquisition

| Parameter | Default | Description |
|-----------|---------|-------------|
| `suopt` | 0 | 1 = read observed data and geometry from SU format (`dat_NNNN.su`); 0 = read from text acquisition file |
| `acquifile` | (suopt=0) | Text file listing source (flag 0) and receiver (flag 1) coordinates: `z x y dip azimuth isrc` |
| `shots` | 1..nproc | Comma-separated list of shot indices, one per MPI rank |
| `nrec_max` | 100000 | Maximum number of receivers per shot (for buffer allocation) |
| `ri` | 4 | Kaiser-windowed sinc interpolation half-width (2*ri+1 points) |

### Data weighting and muting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dxwdat` | 100 | Spatial increment for data weights |
| `xwdat` | 1,1,1,1,1,1 | Data weights as a function of offset (linear interpolation) |
| `muteopt` | 0 | 0=no mute, 1=front mute, 2=tail mute, 3=front+tail mute |
| `ntaper` | 10 | Cosine taper length (points) |
| `xmute1` | (muteopt 1,3) | Offset values for front-mute polygon (m) |
| `tmute1` | (muteopt 1,3) | Time values for front-mute polygon (s) |
| `xmute2` | (muteopt 2,3) | Offset values for tail-mute polygon (m) |
| `tmute2` | (muteopt 2,3) | Time values for tail-mute polygon (s) |

### FWI parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `family` | 1 | 1 = (ln(vp), ln(rho)); 2 = (ln(vp), ln(Ip)) with Ip=rho*vp |
| `npar` | 2 | Number of inversion parameter classes |
| `idxpar` | 1,2 | Which parameter classes to invert: 1=vp, 2=rho/Ip, 3=qinv |
| `niter` | 50 | Maximum number of outer iterations |
| `tol` | 1e-8 | Convergence tolerance (relative to initial misfit) |
| `opt_method` | 1 | 0=NLCG (Polak-Ribière), 1=l-BFGS, 2=Newton-CG |
| `npair` | 5 | l-BFGS memory length |
| `ncg` | 10 | Maximum inner CG iterations (Newton-CG only) |
| `tol_cg` | 0.01 | Relative residual tolerance for Newton-CG |
| `nls` | 20 | Maximum line-search iterations per outer iteration |
| `c1` | 1e-4 | Wolfe condition parameter (sufficient decrease) |
| `c2` | 0.9 | Wolfe condition parameter (curvature) |
| `alpha` | 1.0 | Initial step length (automatically scaled at first gradient) |
| `bound` | 1 | 1=clip parameters by lower/upper bounds |
| `vpmin` / `vpmax` | (bound=1) | Velocity bounds |
| `rhomin` / `rhomax` | (bound=1) | Density bounds |
| `qinvmin` / `qinvmax` | (bound=1) | Attenuation bounds |
| `preco` | 1 | 0=no preconditioner; 1=implicit-reparameterization preconditioner |
| `objopt` | 0 | 0=L2 norm; 1=AWI (not yet implemented) |

### Preconditioner

When `preco=1`, the preconditioner applies:

```text
P = D * [eta*S*S^T + (1-eta)*I] * D
```

where `S` is the triangle smoother, `D` is optional depth weighting, and
`eta = exp(-iter/niter)` decays from 1 to ~0.37 over the run.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `preco_r1` | ~5% of n1 | Smoothing radius in depth |
| `preco_r2` | ~5% of n2 | Smoothing radius in x |
| `preco_r3` | ~5% of n3 | Smoothing radius in y |
| `preco_depth` | 1 | 1=enable depth weighting; 0=uniform |
| `preco_depth_floor` | 0.1 | Minimum depth weight (fraction of surface) |
| `preco_depth_power` | 1.0 | Power-law exponent for depth weighting |

The bathymetry mask is always applied after preconditioning — gradient cells
at or above the water bottom are set to zero.

### Gradient check (mode 4)

Set `checkgrad=1` to run a centered finite-difference verification
immediately after the initial objective and gradient evaluation:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `checkgrad` | 0 | 1=enable gradient check |
| `checkpar` | 1 | 0=all, 1=ln(vp), 2=ln(rho)/ln(Ip), 3=ln(qinv) |
| `checkh` | 0.005 | Finite-difference step size |
| `checkh2` | 0.0025 | Second (smaller) step size for convergence ratio |

The ratio `relerr(checkh)/relerr(checkh2)` should approach 4 in the
second-order regime.

### LSM parameters (mode 3)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lsm_method` | 1 | 1=CG solve of normal equations; 2=FISTA L1-regularized |
| `niter` | (CG) | Number of CG iterations |
| `tol_l1` | 1e-4 | Relative update tolerance for FISTA |
| `lambda_l1` | auto | L1 penalty weight (FISTA only). Default = 0.01*max|J^T W^T W (dobs-d0)| |
| `power_iter` | 6 | Normal-operator power iterations (FISTA, for Lipschitz estimate) |
| `test_tlm_adjoint` | 0 | 1=run TLM/adjoint consistency test instead of solving |

### Source inversion (mode 5)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stffile` | (required) | Output file for estimated source wavelet |
| `source_eps` | 1e-4 | Tikhonov regularisation factor for spectral division |

The source is estimated trace-by-trace in frequency:

```text
stf = argmin sum_{rec} ||W*(g_rec * stf - d_obs)||^2
```

where `g_rec` is the impulse response (Green's function) generated by
`modelling()` with `stf[0]=1`.

### Checkpoint and runtime controls

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nsnap` | 10 | Binomial checkpoint budget for Tapenade reverse. Must be in 1..98 when nt>1 |
| `itcheck` | (unused) | Step at which to write a wavefield snapshot |
| `eachopt` | 0 | 1=read/write per-shot source wavelets (`stffile_NNNN`) |

## Optimisation methods

### NLCG (opt_method=0)

Polak-Ribière nonlinear conjugate gradient with automatic restart when the
`beta` coefficient becomes negative. The preconditioned search direction is:

```text
d_{k+1} = -P*g_{k+1} + beta * d_k
```

### l-BFGS (opt_method=1, default)

Limited-memory BFGS with the two-loop recursion (Nocedal, Algorithm 7.4).
The default memory length is 5 pairs. The initial inverse Hessian is scaled
as `gamma = (s^T y) / (y^T y)`. A Wolfe-condition line search ensures
sufficient decrease and curvature.

### Newton-CG (opt_method=2)

Inexact Newton method with a CG solve for the Gauss-Newton system:

```text
H_GN * d = -g
```

where `H_GN = J^T W^T W J` is applied matrix-free via `lsm_hessian_vector()`.
The inner CG runs up to `ncg` iterations or until the relative residual drops
below `tol_cg`. Preconditioning uses the same implicit-reparameterization
operator as the gradient.

## Linearized waveform inversion

### CG normal equations (lsm_method=1, default)

Solves the unregularised normal equations:

```text
(J^T W^T W J) * dm = J^T W^T W * (dobs - d0)
```

where `d0` is the background data computed by the current model. Each CG
iteration evaluates one TLM (J) and one adjoint (J^T) application.

### FISTA L1 (lsm_method=2)

Solves the L1-regularised problem:

```text
min_dm 0.5 * ||W(J*dm - (dobs-d0))||_2^2 + lambda_l1 * ||dm||_1
```

FISTA applies soft-thresholding after each gradient step with a conservative
step size estimated from normal-operator power iterations. The recovered
perturbation is written as `param_l1_final` and a final TLM evaluation writes
`dcal_l1_final_NNNN` for each shot.

## TLM/adjoint consistency test

Set `test_tlm_adjoint=1` with mode=3 to run a built-in deterministic
consistency check. It compares:
- The tangent-linear derivative against a centered finite difference
- The adjoint directional derivative against the TLM dot product (FWI seed)
- An arbitrary data-seed adjoint identity
- A small-scale normal-operator positive-definiteness check

Example:
```sh
mpirun -n 1 ../bin/SMIwiz mode=3 test_tlm_adjoint=1 acquifile=acqui.txt \
  vpfile=vp_init rhofile=rho_init stffile=fricker nt=100 dt=0.0005 \
  n1=41 n2=41 d1=5 d2=5 nb=20 order=4 freesurf=0
```

## Source encoding

Each MPI rank processes one shot. The shot-to-rank mapping is determined by
the `shots` parameter (a comma-separated list of length nproc). If omitted,
shots are assigned as `shot_idx[iproc] = iproc+1` (1-indexed).

- **Binary data I/O**: Each rank reads `dat_NNNN` (observed data) and writes
  `dsyn_NNNN` (synthetic data) and `dres_NNNN` (residual).
- **SU data I/O**: With `suopt=1`, each rank reads `dat_NNNN.su` and extracts
  trace headers for source/receiver geometry.
- **Mask**: Each rank writes `mask_NNNN` (the `wdat` weighting array).

## Output files

| File | Mode | Description |
|------|------|-------------|
| `dat_NNNN` | 0 | Synthetic shot records (binary float, nt×nrec) |
| `dsyn_NNNN` | 1,4 | Synthetic data at current model |
| `dres_NNNN` | 1,4 | Data residual (dcal - dobs) |
| `mask_NNNN` | 1,2,3,4 | Data-weighting mask (nt×nrec) |
| `gradient_fwi` | 1 (first iter), 2 | FWI gradient or RTM image (log-parameter space) |
| `param_final` | 1,3 | Recovered log-parameters |
| `param_iter` | 1 | Intermediate parameters at each iteration (appended) |
| `gradient_final` | 1 | Gradient at convergence |
| `gradient_iter` | 1 | Intermediate gradients (appended) |
| `iterate.txt` | 1 | Convergence history |
| `d0_NNNN` | 3 | Background data (unperturbed) |
| `param_l1_final` | 3 (lsm_method=2) | L1-FISTA recovered perturbation |
| `param_l1_iter` | 3 (lsm_method=2) | L1-FISTA iteration history |
| `dcal_l1_final_NNNN` | 3 (lsm_method=2) | Final linearized synthetic (d0 + J*dm) |
| `image_rtm` | 2 | RTM image (same file as gradient_fwi in mode=2) |
| `time_info.txt` | 0 | Forward-modelling timing breakdown |
| `stffile` / `stffile_NNNN` | 5 | Estimated source wavelet |

## Run examples

### Forward modelling (mode=0)
```sh
mpirun -n 2 ../bin/SMIwiz mode=0 acquifile=acqui.txt vpfile=vp rhofile=rho \
  stffile=fricker nt=2000 dt=0.0005 n1=141 n2=141 d1=5 d2=5 nb=20
```

Viscoacoustic forward modelling:
```sh
mpirun -n 2 ../bin/SMIwiz mode=0 acquifile=acqui.txt vpfile=vp rhofile=rho \
  stffile=fricker qinvfile=qinv_init nt=2000 dt=0.0005 n1=141 n2=141 \
  d1=5 d2=5 nb=20 nrelax=3 Q0=100 sls_fmin=1 sls_fmax=40
```

### FWI (mode=1)
```sh
cd run_fwi2d
./run.sh
```

This runs 50 l-BFGS iterations with two shots, family-1 (ln(vp), ln(rho)).
See `run_fwi2d_GN/` for a Newton-CG variant (`opt_method=2`).

### RTM (mode=2)
```sh
cd run_rtm2d
./run.sh
```

Uses the FWI gradient kernel (mode=2 in `do_fwi()`) to form an impedance-
contrast image. The output is written to `image_rtm`.

### Linearized inversion (mode=3)
```sh
cd run_lsm2d
./run.sh
```

CG solve of the normal equations. For the Marmousi LSRTM example:
```sh
cd run_lsrtm2d_marmousi
./run_lsrtm.sh
```

### Gradient check (mode=4)
```sh
cd run_gradient_check
./run.sh
PARFILE=inputpar_family2.txt ./run.sh
PARFILE=inputpar_qinv.txt ./run.sh
```

Each run performs one FWI gradient evaluation immediately followed by a
second-order centered finite-difference check for the selected parameter
class.

### Source inversion (mode=5)
```sh
mpirun -n 2 ../bin/SMIwiz mode=5 acquifile=acqui.txt vpfile=vp rhofile=rho \
  stffile=estimated_source nt=2000 dt=0.0005 n1=141 n2=141 d1=5 d2=5 \
  nb=20 stffile=estimated_source source_eps=1e-4
```

The impulse response is computed with `stf[0]=1.0` then the source is
estimated trace-by-trace in the frequency domain with Tikhonov regularisation.

## Run scripts

The `run_gradient_check/run.sh` script accepts environment variables for
viscoacoustic parameterisation:

```sh
NRELAX=3 Q0=100 SLS_FMIN=1 SLS_FMAX=40 ./run.sh
```

All run scripts respect these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SMIWIZ_BIN` | `../bin/SMIwiz` | Path to the executable |
| `NP` | (varies) | Number of MPI ranks |
| `MPIEXEC` | `mpirun` | MPI launcher; set to `direct` to skip the launcher |
| `PARFILE` | `inputpar.txt` | Parameter file to use |
| `NRELAX` | (unset) | Override `nrelax` |
| `Q0` | (unset) | Override target Q |
| `SLS_FMIN` | (unset) | Override SLS minimum frequency |
| `SLS_FMAX` | (unset) | Override SLS maximum frequency |

## FDTD kernel variants

Two FDTD implementations are available and compiled together:

**`fdtd.c`** (original): The innermost loops carry PML-boundary branches
(`if(i1<sim->nb)`, etc.) and `sim->order` tests at every grid point. This is
the version seen by Tapenade and is the one differentiated.

**`fdtd_split.c`** (optimised): Splits each FDTD step into one interior loop
(no PML branches, coefficients pre-loaded from arrays) followed by six
boundary-face loops covering the PML layers where branches are perfectly
predictable. This version improves auto-vectorisation and branch prediction
for large models but is **not** seen by Tapenade. The linker resolves
`fdtd_update_v()` and `fdtd_update_p()` from `fdtd.c` for the AD sources
and from `fdtd_split.c` for all other calls.

## Copyright

Copyright (c) Pengliang Yang, 2026, Laoshan Laboratory, China

Homepage: https://yangpl.wordpress.com

E-mail: ypl.2100@gmail.com
