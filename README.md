# SMIwiz_visco — Seismic Modelling, RTM Imaging, and Full Waveform Inversion

This checkout builds the MPI application `bin/SMIwiz` — an integrated toolbox
for 2D/3D seismic forward modelling, reverse-time migration (RTM), linearized
and nonlinear full waveform inversion (FWI), and source time-function
estimation.  The forward propagator supports an isotropic acoustic step and an
optional three-standard-linear-solid (SLS) viscoacoustic step. Algorithmic
differentiation (AD) via Tapenade 3.16 generates the tangent-linear and adjoint
routines that supply the gradient for FWI and the Jacobian operators for
least-squares reverse time migration (LSRTM).

## Features

- **2D and 3D** isotropic acoustic and viscoacoustic finite-difference modelling
  on a staggered grid (4th- or 8th-order spatial stencils).
- **CPML** absorbing boundary conditions with a stress-free surface option.
- **Kaiser-windowed sinc interpolation** for source injection and receiver
  extraction (off-grid positioning).
- **Three-parameter memory-variable viscoacoustics** (SLS mechanisms) fitted
  to a constant-Q target over a user-specified frequency band.
- **Tapenade-generated adjoint** (`modelling_b`) and tangent-linear
  (`modelling_d`) kernels for exact discrete gradients.
- **Nonlinear FWI** with three optimization methods:
  - `opt_method=0`: NLCG (nonlinear conjugate gradient)
  - `opt_method=1`: l-BFGS (limited-memory BFGS; default)
  - `opt_method=2`: Newton-CG (inner CG solve of the Gauss-Newton system).
- **Two parameter families**:
  - `family=1`: (ln vp, ln rho)
  - `family=2`: (ln vp, ln ip) where ip = vp × rho (impedance).
- **Selective parameter activation** via `idxpar`: any subset of {vp, rho/ip, qinv}.
- **Bathymetry masking** — zero gradient in the water column.
- **Implicit reparameterization preconditioner** — separable triangle smoothing
  with depth weighting, η annealing from smooth to identity across iterations.
- **Data-domain linearized inversion** (LSM / LSRTM) using the generated
  tangent and adjoint operators for matrix-free CG or L1-FISTA solves.
- **Source wavelet estimation** in the frequency domain (Pratt, 1999).
- **SU-format data I/O** (`suopt=1`) for field data processing.
- **Data muting and weighting** — offset-dependent weights, front/tail mute
  with cosine taper.
- **Per-shot source wavelets** (`eachopt=1`) for multi-source experiments.
- **Gradient checking** (mode=4) for validation of AD gradients.
- **OpenMP parallelism** inside each MPI rank for the stencil computation.

## Runtime modes

The executable supports these `mode` values:

| Mode | Description |
|------|-------------|
| 0 | Forward modelling — synthetic shot records from the supplied model |
| 1 | Nonlinear FWI — iterative data misfit minimization (NLCG / l-BFGS / Newton-CG) |
| 2 | RTM — reverse-time migration (impedance-gradient image via the FWI kernel) |
| 3 | Linearized waveform inversion (LSM / LSRTM) — CG or L1-FISTA using the Born operator |
| 4 | Standalone FWI gradient — single objective/gradient evaluation (for validation) |
| 5 | Source wavelet inversion — frequency-domain estimation from observed data |

## Requirements

- An **MPI** implementation with C development support. `mpicc` must compile the
  sources; `mpirun` launches multi-rank runs.
- **FFTW3** development headers and library. The prefix must contain
  `include/fftw3.h` and `lib/libfftw3.so`.
- **Tapenade 3.16** with a compatible Java runtime. `TAPENADE_HOME` must contain
  `bin/tapenade` and `ADFirstAidKit/` (provides `adStack.c` and `adBinomial.c`).
- **Python 3** with `numpy` and `matplotlib` for run-scripts and convergence plots.
- **SU** may be needed to visualize binary files (generate Ricker wavelet)

On Debian/Ubuntu:

```sh
sudo apt install openmpi-bin libopenmpi-dev libfftw3-dev default-jre
```

Install Tapenade separately, then verify the toolchain:

```sh
mpicc --version
test -f /usr/include/fftw3.h
test -x /path/to/tapenade_3.16/bin/tapenade
test -d /path/to/tapenade_3.16/ADFirstAidKit
```

## Build

From the repository root:

```sh
make -C src
```

The default target removes `src/tapenade_out/`, regenerates the Tapenade
reverse and tangent sources, repairs the generated C code, then compiles and
links `bin/SMIwiz`.  Run `make -C src clean` to remove build products and
generated Tapenade files only (no source cleanup).

For a non-default installation layout:

```sh
make -C src fftw3=/path/to/fftw3 TAPENADE_HOME=/path/to/tapenade_3.16
```

### Build targets

| Target | Effect |
|--------|--------|
| `all` (default) | Clean, regenerate AD sources, build `bin/SMIwiz` |
| `adjoint-generate` | Run Tapenade reverse mode on the primal sources only |
| `tlm-generate` / `tangent-generate` | Run Tapenade tangent-linear mode only |
| `clean` | Remove object files, `bin/SMIwiz`, and `tapenade_out/` |

Parallel builds (`make -j`) are supported. The `adjoint-generate` and
`tlm-generate` targets have a shared dependency on the preprocessed primal
sources and are serialised by the Makefile order-only prerequisites.

## Source tree

```
├── bin/               Compiled executable (SMIwiz)
├── doc/               Documentation (e.g. Q-fitting validation PDF)
├── include/
│   ├── acq.h          Acquisition geometry (sources, receivers, interpolation)
│   ├── cstd.h         Common types, memory management, parameter parsing
│   ├── fwi.h          FWI state (family, bounds, iteration tracking)
│   ├── modelling.h    Forward / adjoint / tangent modelling interfaces
│   ├── opt.h          Optimiser state (l-BFGS, line search, NLCG, Newton-CG)
│   ├── segy.h         SU/SEG-Y trace header struct
│   └── sim.h          Simulation state (grid, fields, wavefields, FDTD arrays)
├── Marmousi/          Marmousi benchmark model (vp.bin, rho.bin)
├── src/
│   ├── Makefile            Build rules and Tapenade AD pipeline
│   ├── main.c              MPI initialisation, parameter parsing, mode dispatch
│   ├── acq_init_free.c     Acquisition geometry initialisation / cleanup
│   ├── modelling.c         Top-level forward modelling driver
│   ├── do_modelling.c      Mode=0 wrapper
│   ├── do_fwi.c            Mode=1,4 wrapper — FWI loop (NLCG / l-BFGS / Newton-CG)
│   ├── do_rtm.c            Mode=2 wrapper — RTM (delegates to do_fwi)
│   ├── do_lsm.c            Mode=3 — linearized waveform inversion (CG / L1-FISTA)
│   ├── do_invert_source.c  Mode=5 — source wavelet estimation
│   ├── fg_fwi.c            Objective and gradient evaluation (Tapenade adjoint driver)
│   ├── gradient_check.c    Centered finite-difference gradient validation
│   ├── fdtd.c              Forward FDTD core (staggered-grid pressure-velocity)
│   ├── fdtd_split.c        Forward/adjoint FDTD with loop splitting for Tapenade
│   ├── cpml.c              CPML absorbing boundary coefficients
│   ├── extend_model.c      Model padding (add CPML layers)
│   ├── inject_extract.c    Source injection / receiver extraction (Kaiser sinc)
│   ├── read_write_data.c   Binary and SU data I/O
│   ├── setup_data_mask.c   Data weighting and muting
│   ├── precondition.c      Implicit reparameterization preconditioner
│   ├── smoothing.c         Separable triangle smoothing (Claerbout)
│   ├── opt.c               l-BFGS, NLCG, and line-search implementation
│   ├── visco_lsq.c         Constant-Q SLS fit (Levenberg-Marquardt)
│   ├── kaiser_windowed_sinc.c  Kaiser-Bessel window for sinc interpolation
│   ├── check_cfl.c         CFL stability check
│   └── fix_tapenade_c.sh   Post-processing fixes for generated AD code
├── run_fwi2d/          Example: 2D FWI with Marmousi (mode=1)
└── README.md
```

## Parameter reference

The executable reads parameters as `key=value` pairs from the command line or a
parameter file (via `$(cat inputpar.txt)` shell expansion).

### Grid and time

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n1` | required | Grid points in 1st dimension (depth) |
| `n2` | required | Grid points in 2nd dimension (crossline) |
| `n3` | 1 | Grid points in 3rd dimension (inline); >1 activates 3D |
| `d1` | required | Grid spacing in 1st dimension (m) |
| `d2` | required | Grid spacing in 2nd dimension (m) |
| `d3` | (with n3>1) | Grid spacing in 3rd dimension (m) |
| `nt` | required | Number of time steps |
| `dt` | required | Temporal sampling (s) |
| `order` | 4 | FD order (4 or 8) |
| `nb` | 20 | Number of CPML padding layers |

### Physics

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vpfile` | required | Binary file of P-wave velocity (n1×n2×n3 float) |
| `rhofile` | required | Binary file of density |
| `qinvfile` | (with nrelax>0) | Binary file of inverse Q (1/Q) |
| `stffile` | (mode≠5) | Binary file of source time function (nt float) |
| `nrelax` | 0 | 0=acoustic; 3=three-mechanism viscoacoustic |
| `Q0` | 100 | Target Q for SLS fitting (used with nrelax=3) |
| `sls_fmin` | 1 | SLS minimum frequency (Hz) |
| `sls_fmax` | 40 | SLS maximum frequency (Hz) |
| `freesurf` | 1 | 1=stress-free surface; 0=no free surface |
| `freq` | 15 | Reference frequency for CPML (Hz) |
| `eachopt` | 0 | 1=per-shot source wavelet (stffile_NNNN); 0=shared stffile |

### Acquisition

| Parameter | Default | Description |
|-----------|---------|-------------|
| `suopt` | 0 | 1=SU-format data; 0=text acquisition file |
| `acquifile` | (suopt=0) | Acquisition geometry file |
| `shots` | 1..nproc | Comma-separated shot indices, one per MPI rank |
| `zmin` / `zmax` | 0 / model extent | Model coordinate range |
| `xmin` / `xmax` | 0 / model extent | Model coordinate range |
| `ymin` / `ymax` | 0 / model extent | Model coordinate range |

### FWI and optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `opt_method` | 1 | 0=NLCG; 1=l-BFGS; 2=Newton-CG |
| `niter` | 50 | Maximum outer iterations |
| `nls` | 20 | Maximum line-search steps per iteration |
| `npair` | 5 | l-BFGS memory length (correction pairs) |
| `ncg` | 5 | Inner CG iterations for Newton-CG |
| `tol` | 1e-8 | Convergence tolerance on gradient norm |
| `c1` | 1e-4 | Wolfe condition (sufficient decrease) |
| `c2` | 0.9 | Wolfe condition (curvature) |
| `alpha` | 1.0 | Initial step length |
| `bound` | 1 | 1=enforce parameter bounds |
| `preco` | 1 | 1=apply preconditioner |
| `family` | (required) | 1=ln(vp),ln(rho); 2=ln(vp),ln(ip) |
| `npar` | (required) | Number of inversion parameters |
| `idxpar` | (required) | Comma-separated: 1=vp, 2=density/impedance, 3=qinv |
| `vpmin` / `vpmax` | — | Velocity bounds (m/s) |
| `rhomin` / `rhomax` | — | Density bounds |
| `qinvmin` / `qinvmax` | — | Attenuation bounds |
| `objopt` | 1 | 0=L2; 1=AWI (adaptive waveform inversion) |
| `nsnap` | 20 | Binomial checkpoint budget for Tapenade reverse |
| `restart` | 0 | Restart iteration count |

### Preconditioner

| Parameter | Default | Description |
|-----------|---------|-------------|
| `preco_r1` | 5% of n1 | Triangle smoothing radius in 1st dimension |
| `preco_r2` | 5% of n2 | Triangle smoothing radius in 2nd dimension |
| `preco_r3` | 5% of n3 | Triangle smoothing radius in 3rd dimension |
| `preco_depth` | 1 | 1=enable depth weighting |
| `preco_depth_floor` | 0.1 | Minimum depth weight |
| `preco_depth_power` | 1.0 | Depth-weight power |

### Data weighting and muting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dxwdat` | 100 | Offset interval for data weights |
| `xwdat` | 1,1,1,1,1,1 | Offset-dependent weights (linear interpolation) |
| `muteopt` | 0 | 0=no mute; 1=front mute; 2=tail mute; 3=front+tail |
| `ntaper` | 10 | Cosine taper length (samples) |
| `xmute1` | — | Offset breakpoints for front mute (m) |
| `tmute1` | — | Time breakpoints for front mute (s) |
| `xmute2` | — | Offset breakpoints for tail mute (m) |
| `tmute2` | — | Time breakpoints for tail mute (s) |

## Data objective

The scalar misfit minimized in modes 1 and 4 is:

```text
J = 0.5 × sum_{receiver,time} [wdat × (dcal − dobs)]²
```

`modelling()` is the nonlinear map from the physical model to `dcal`. It does
not receive observations, data weights, or a scalar objective.  Tapenade
differentiates `dcal` with respect to the physical `vp`, `rho`, and `qinv`
inputs.  `fg_fwi()` accumulates J in double precision, seeds the reverse
sweep with `dcalb = wdat² × (dcal − dobs)`, and maps the returned
physical-model derivatives to the selected logarithmic parameters.  Because
the objective is outside the differentiated call graph, this normalization
changes the external reverse seed but not the Tapenade head or the generated
Jacobian-transpose implementation.

## Input/output files

Files are written in the working directory of each run.  `NNNN` is the shot
index (zero-padded to four digits).

| File | Mode(s) | Description |
|------|---------|-------------|
| `dat_NNNN` | 0 | Synthetic shot records (binary float, nt × nrec) |
| `dsyn_NNNN` | 1, 4 | Synthetic data at current model |
| `dres_NNNN` | 1, 4 | Data residual (dcal − dobs) |
| `mask_NNNN` | 1, 2, 3, 4 | Data-weighting mask (nt × nrec) |
| `gradient_fwi` | 1 (iter 0), 2 | FWI gradient or RTM image (log-parameter space) |
| `param_final` | 1, 3 | Recovered log-parameters |
| `param_iter` | 1 | Intermediate log-parameters at each iteration (appended) |
| `gradient_final` | 1 | Gradient at convergence |
| `gradient_iter` | 1 | Intermediate gradients (appended) |
| `iterate.txt` | 1 | Convergence history (iteration, misfit, step) |
| `d0_NNNN` | 3 | Background (unperturbed) synthetic data |
| `param_l1_final` | 3 (L1-FISTA) | L1-FISTA recovered perturbation |
| `param_l1_iter` | 3 (L1-FISTA) | L1-FISTA iteration history |
| `dcal_l1_final_NNNN` | 3 (L1-FISTA) | Final linearized synthetic (d₀ + J·dm) |
| `image_rtm` | 2 | RTM image (same output as gradient_fwi in mode=2) |
| `time_info.txt` | 0 | Forward-modelling timing breakdown |
| `stffile` / `stffile_NNNN` | 5 | Estimated source wavelet |
| `snapshot.bin` | 0 | Wavefield snapshot at `itcheck` |

## Run examples

Parameters are supplied as command-line `key=value` pairs, piped through
`$(cat inputpar.txt)` in the run scripts.

### Forward modelling (mode=0)

Acoustic:
```sh
mpirun -n 2 ../bin/SMIwiz mode=0 acquifile=acqui.txt vpfile=vp rhofile=rho \
  stffile=fricker nt=2000 dt=0.0005 n1=141 n2=141 d1=5 d2=5 nb=20
```

Viscoacoustic (three SLS mechanisms):
```sh
mpirun -n 2 ../bin/SMIwiz mode=0 acquifile=acqui.txt vpfile=vp rhofile=rho \
  stffile=fricker qinvfile=qinv_init nt=2000 dt=0.0005 n1=141 n2=141 \
  d1=5 d2=5 nb=20 nrelax=3 Q0=100 sls_fmin=1 sls_fmax=40
```

### FWI (mode=1)

The `run_fwi2d` directory provides a complete 2D Marmousi FWI example using
l-BFGS (opt_method=1) with two logarithmic parameters (vp and rho):

```sh
cd run_fwi2d
./run.sh
```

This generates synthetic observations from initial models, writes the parameter
file `inputpar.txt`, and launches the inversion.  The example uses 50
iterations, 24 MPI ranks, and the Marmousi starting models.

Substitute `opt_method=0` for NLCG or `opt_method=2` with `ncg=5` for
Newton-CG in the parameter file.

### RTM (mode=2)

```sh
mpirun -n 24 ../bin/SMIwiz mode=2 acquifile=acqui.txt vpfile=vp \
  rhofile=rho stffile=fricker nt=2500 dt=0.002 n1=151 n2=461 d1=20 d2=20 nb=20
```

Uses the same FWI impedance-gradient kernel to form an RTM image.  The output
is written to `image_rtm`.

### Linearized inversion — LSM / LSRTM (mode=3)

```sh
mpirun -n 24 ../bin/SMIwiz mode=3 acquifile=acqui.txt vpfile=vp_init \
  rhofile=rho_init stffile=fricker nt=2500 dt=0.002 n1=151 n2=461 \
  d1=20 d2=20 nb=20 niter=30 family=2 npar=2 idxpar=1,2
```

Solves the normal equations JᵀWᵀWJ·dm = JᵀWᵀW(dobs − d₀) via CG, or the
L1-regularized problem with FISTA (`lsm_method=2`), using the generated
tangent and adjoint operators — no hand-maintained Born propagator.

### Gradient check (mode=4)

```sh
mpirun -n 1 ../bin/SMIwiz mode=4 acquifile=acqui.txt vpfile=vp \
  rhofile=rho stffile=fricker nt=2500 dt=0.002 n1=151 n2=461 \
  d1=20 d2=20 nb=20 family=1 npar=1 idxpar=1 \
  checkgrad=1 checkpar=1 checkh=1e-4 checkh2=1e-6
```

Performs one FWI gradient evaluation followed by a second-order centered
finite-difference check for the specified parameter class.  The `checkpar`
value selects the parameter index (1=vp, 2=rho/ip, 3=qinv).  `checkh` and
`checkh2` control the perturbation step for the FD check.

### Source inversion (mode=5)

```sh
mpirun -n 2 ../bin/SMIwiz mode=5 acquifile=acqui.txt vpfile=vp rhofile=rho \
  stffile=estimated_source nt=2000 dt=0.0005 n1=141 n2=141 d1=5 d2=5 \
  nb=20 source_eps=1e-4
```

The impulse response is computed with `stf[0] = 1.0`, then the source is
estimated trace-by-trace in the frequency domain with Tikhonov regularisation.

## Run script conventions

All run scripts respect these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SMIWIZ_BIN` | `../bin/SMIwiz` | Path to the executable |
| `NP` | (varies) | Number of MPI ranks |
| `MPIEXEC` | `mpirun` | MPI launcher; set to `direct` to skip it |
| `PARFILE` | `inputpar.txt` | Parameter file |
| `NRELAX` | (unset) | Override `nrelax` |
| `Q0` | (unset) | Override target Q |
| `SLS_FMIN` | (unset) | Override SLS minimum frequency |
| `SLS_FMAX` | (unset) | Override SLS maximum frequency |

## Data format notes

- **Binary float I/O**: All model and data binary files are native-endian
  32-bit IEEE floats, stored as a contiguous array in Fortran-like
  (z-fastest, then x, then y) order.
- **SU data I/O**: With `suopt=1`, each rank reads `dat_NNNN.su` and extracts
  trace headers for source/receiver geometry.  The number of receivers is
  determined from the file size.

## Copyright

Copyright (c) Pengliang Yang, 2026, Laoshan Laboratory, China

Homepage: https://yangpl.wordpress.com

E-mail: ypl.2100@gmail.com
