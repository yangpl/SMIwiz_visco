#!/usr/bin/env python3
"""
Resample Marmousi model (vp.bin, rho.bin) from 751×2301 to 151×461
by smoothing (Gaussian anti-alias filter) and taking every 5th sample.

Usage: python resample_marmousi.py
"""

import numpy as np
from scipy.ndimage import gaussian_filter

# Input / output parameters
nx, nz = 751, 2301          # original grid size (x, z)
step = 5                     # downsampling factor
nx_out, nz_out = 151, 461   # target grid size
sigma = 5                  # Gaussian filter sigma (pixels); step/2 ≈ conservative

files = [
    ("vp.bin",  "vp_true.bin"),
    ("rho.bin", "rho_true.bin"),
]

for in_name, out_name in files:
    # Read original binary as float32
    data = np.fromfile(in_name, dtype=np.float32).reshape((nx, nz), order='F')

    # Anti-alias filter before downsampling
    data = gaussian_filter(data, sigma=sigma, mode='reflect')

    # Downsample: take every 'step'-th element
    data_ds = data[::step, ::step]   # shape (151, 461) with step=5

    # Verify size
    assert data_ds.shape == (nx_out, nz_out), \
        f"Expected ({nx_out}, {nz_out}), got {data_ds.shape}"

    # Write as binary (column-major to match original layout)
    data_ds.ravel(order='F').astype(np.float32).tofile(out_name)

    print(f"{in_name:10s}  {data.shape!r:20s} → {data_ds.shape!r:20s}  {out_name}")

print("")
print("--- Smoothing true files to create initial models ---")

# Heavier smoothing on the true (151×461) grid to produce initial models
init_sigma = 4.0              # smoothing on the coarse grid

init_files = [
    ("vp_true.bin", "vp_init.bin"),
    ("rho_true.bin", "rho_init.bin"),
]

for in_name, out_name in init_files:
    data = np.fromfile(in_name, dtype=np.float32).reshape((nx_out, nz_out), order='F')
    data_smooth = gaussian_filter(data, sigma=init_sigma, mode='reflect')
    data_smooth.ravel(order='F').astype(np.float32).tofile(out_name)
    print(f"{in_name:20s} → {out_name:20s}  sigma={init_sigma}")

print("")
print("--- Creating inverse-Q (attenuation) model from smoothed Vp ---")

# Empirical Qp-Vp relationship:  Qp = 10 + 0.025 * Vp  (Vp in m/s)
# Gives reasonable Qp range:  ~48 at 1500 m/s,  ~148 at 5500 m/s
# Clamp to [10, 999] to avoid unrealistically high values
# Then output inverse Q (1/Qp) so the FDTD code can use it directly

vp = np.fromfile("vp_true.bin", dtype=np.float32).reshape((nx_out, nz_out), order='F')
qp = 10.0 + 0.035 * vp
qp = np.clip(qp, 10.0, 999.0)
qinv = 1.0 / qp
qinv.ravel(order='F').astype(np.float32).tofile("qinv_true.bin")
print(f"vp_true.bin → qinv_true.bin  1/Qp range: {qinv.min():.6f}–{qinv.max():.6f}")

print("")
print("--- Creating constant inverse-Q initial model ---")

qinv_const = np.full((nx_out, nz_out), 0.01, dtype=np.float32)   # corresponds to Qp = 100
qinv_const.ravel(order='F').astype(np.float32).tofile("qinv_init.bin")
print(f"{'constant':20s} → qinv_init.bin        1/Qp = 0.01 (Qp = 100) everywhere")
