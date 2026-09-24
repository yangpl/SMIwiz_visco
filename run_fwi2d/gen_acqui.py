#!/usr/bin/env python3
"""
Generate acqui.txt for the Marmousi model with surface-only acquisition.

Each shot gather: one source line followed by all receiver lines.
Receivers are identical for every source (surface array).
28 sources evenly placed from 100 m to 9100 m.

Format matches the reference in ../run_lsrtm2d/acqui.txt.

Usage:
  python3 gen_acqui.py                     # default → acqui.txt
  python3 gen_acqui.py -o my_acqui.txt     # custom name
"""

import argparse
import numpy as np

# ============================================================
#  CONFIGURATION
# ============================================================
MODEL_NZ = 151
MODEL_NX = 461
DZ = 20.0
DX = 20.0

SRC_Z = 5.0                # source depth (m)
RECV_Z = 10.0               # receiver depth (m)  — just below surface

SRC_X_MIN = 100.0           # first source x (m)
SRC_X_MAX = 9100.0          # last source x (m)
N_SRC = 24                  # number of sources

# ============================================================
#  Derived
# ============================================================
X_MIN = 0.0
X_MAX = (MODEL_NX - 1) * DX   # 9200 m

# Receiver x-positions: every grid point from 0 to 9200
recv_xs = np.arange(X_MIN, X_MAX - DX / 2, DX)   # 0, 20, 40, ..., 9200

# Source x-positions: N_SRC evenly spaced from SRC_X_MIN to SRC_X_MAX
src_xs = np.linspace(SRC_X_MIN, SRC_X_MAX, N_SRC)

# Receiver lines (flag=1) — shared by every source
recv_lines = []
for rx in recv_xs:
    recv_lines.append(f"   {RECV_Z:>10.8f}       {rx:>10.6f}       0.00000000       0.00000000       0.00000000               1")


# ============================================================
#  Write
# ============================================================
def write_acqui(output_path: str, src_z: float, src_xs, recv_lines, n_recv: int):
    lines = [
        "z     x    y     azimuth    dip    src/rec(0/1)",
    ]

    for sx in src_xs:
        # Source line
        lines.append(
            f"   {src_z:>10.8f}       {sx:>10.6f}       0.00000000       0.00000000       0.00000000               0"
        )
        # Receiver block
        lines.extend(recv_lines)

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    n_src = len(src_xs)
    print(f"Written {output_path}:")
    print(f"  Sources:   {n_src}  (z={src_z} m, x=[{src_xs[0]:.1f}, {src_xs[-1]:.1f}] m)")
    print(f"  Receivers: {n_recv} per source (z={RECV_Z} m, x=[{X_MIN:.0f}, {X_MAX:.0f}] m)")
    print(f"  Total lines: {1 + n_src * (1 + n_recv)}  (header + {n_src} shots × {1 + n_recv} lines)")


# ============================================================
#  Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="acqui.txt",
                        help="Output file (default: acqui.txt)")
    parser.add_argument("--src-z", type=float, default=SRC_Z,
                        help=f"Source depth (default: {SRC_Z})")
    parser.add_argument("--recv-z", type=float, default=RECV_Z,
                        help=f"Receiver depth (default: {RECV_Z})")
    parser.add_argument("--src-min", type=float, default=SRC_X_MIN,
                        help=f"First source x (default: {SRC_X_MIN})")
    parser.add_argument("--src-max", type=float, default=SRC_X_MAX,
                        help=f"Last source x (default: {SRC_X_MAX})")
    parser.add_argument("--n-src", type=int, default=N_SRC,
                        help=f"Number of sources (default: {N_SRC})")
    parser.add_argument("--nz", type=int, default=MODEL_NZ)
    parser.add_argument("--nx", type=int, default=MODEL_NX)
    parser.add_argument("--dz", type=float, default=DZ)
    parser.add_argument("--dx", type=float, default=DX)
    args = parser.parse_args()

    # Override from CLI
    MODEL_NZ = args.nz
    MODEL_NX = args.nx
    DZ = args.dz
    DX = args.dx
    SRC_Z = args.src_z
    RECV_Z = args.recv_z
    SRC_X_MIN = args.src_min
    SRC_X_MAX = args.src_max
    N_SRC = args.n_src

    X_MAX = (MODEL_NX - 1) * DX

    # Rebuild receiver positions with potentially updated DX/NX
    recv_xs = np.arange(0.0, X_MAX + DX / 2, DX)
    recv_lines = []
    for rx in recv_xs:
        recv_lines.append(
            f"   {RECV_Z:>10.8f}       {rx:>10.6f}       0.00000000       0.00000000       0.00000000               1"
        )

    src_xs = np.linspace(SRC_X_MIN, SRC_X_MAX, N_SRC)

    write_acqui(args.output, SRC_Z, src_xs, recv_lines, len(recv_lines))
