#!/usr/bin/env python3
"""Plot the two-parameter Marmousi l-BFGS result used in Section 5.2.

Writes PNG and PDF versions. Binary models are float32 with depth varying
fastest; param_final contains physical Vp followed by density (not logarithms).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NX, NZ = 461, 151
DX, DZ = 20.0, 20.0
RUN = Path(__file__).resolve().parent


def read_fortran(path: Path, nz: int = NZ, nx: int = NX) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size != nz * nx or not np.isfinite(values).all():
        raise ValueError(f"{path}: expected {nz * nx} finite float32 values")
    return values.reshape(nx, nz).T


def read_param_final(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.fromfile(path, dtype=np.float32)
    if values.size != 2 * NX * NZ or not np.isfinite(values).all():
        raise ValueError(f"{path}: expected two finite fields of size {NZ} x {NX}")
    vp, rho = values.reshape(2, NX, NZ).transpose(0, 2, 1)
    return vp, rho


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marmousi", type=Path, default=RUN.parent / "Marmousi")
    parser.add_argument("--inverted", type=Path, default=RUN / "param_final")
    parser.add_argument("--output", type=Path, default=RUN / "models.png",
                        help="output name; both .png and .pdf are written")
    args = parser.parse_args()
    final = read_param_final(args.inverted)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 2, figsize=(12, 7.5), layout="constrained")
    titles = ("True", "Initial", "Inverted")
    specs = (("vp", "Velocity (m/s)", 1.0, "viridis"),
             ("rho", r"Density (kg/m$^3$)", 1.0, "viridis"))
    for col, (key, label, scale, cmap) in enumerate(specs):
        models = [read_fortran(args.marmousi / f"{key}_{stage}.bin")
                  for stage in ("true", "init")] + [final[col]]
        limits = (min(float(m.min()) for m in models) * scale,
                  max(float(m.max()) for m in models) * scale)
        for row, model in enumerate(models):
            ax = axes[row, col]
            im = ax.imshow(model * scale, origin="upper", cmap=cmap,
                           vmin=limits[0], vmax=limits[1], interpolation="nearest",
                           extent=(0, (NX-1)*DX/1000, (NZ-1)*DZ/1000, 0),
                           aspect="equal")
            field = "velocity" if key == "vp" else "density"
            ax.set_title(f"({chr(97 + 2*row + col)}) {titles[row]} {field}", loc="left")
            ax.set_xlabel("Distance (km)")
            ax.set_xticks([0, 2, 4, 6, 8])
            ax.set_yticks([0, 1, 2, 3])
            if col == 0:
                ax.set_ylabel("Depth (km)")
        fig.colorbar(im, ax=axes[:, col], orientation="horizontal",
                     fraction=0.045, pad=0.035, aspect=35, label=label)
        errors = [float(np.sqrt(np.mean((m.astype(float)-models[0])**2)))
                  for m in models[1:]]
        print(f"{key} RMSE, initial -> final: {errors[0]:.3f} -> {errors[1]:.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        output = args.output.with_suffix(suffix)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Wrote {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
