#!/usr/bin/env python3
"""Plot the Marmousi l-BFGS objective and gradient histories for Section 5.2.

The driver logs iterate.txt before each update and saves param_final and
gradient_final after it. Recover the last accepted objective from out.log
and the last gradient norm from gradient_final to include update 50.
Writes both PNG and PDF. Optionally overlay the Gauss--Newton run with
--compare-run-dir ../run_fwi2d_gn.
For the original SMIwiz comparison, use --run-label Tapenade
--compare-run-dir ../run_fwi2d_lbfgs_SMIwiz --compare-label 'Original SMIwiz'.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path(__file__).resolve().parent


def read_iterations(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 7:
            continue
        try:
            rows.append([float(value) for value in fields])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no iteration records found in {path}")
    data = np.asarray(rows)
    if not np.isfinite(data).all() or data[0, 1] <= 0 or data[0, 3] <= 0:
        raise ValueError(f"invalid iteration records in {path}")
    return data


def convergence_history(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = run / "iterate.txt"
    data = read_iterations(path)
    iterations, objective = data[:, 0], data[:, 2]
    gradient = data[:, 3] / data[0, 3]
    # Only append a final update for a completed run with matching saved models.
    if "==>Maximum iteration number reached!" in path.read_text():
        nvalues = 2 * 151 * 461
        final = np.fromfile(run / "param_final", dtype=np.float32)
        gfinal = np.fromfile(run / "gradient_final", dtype=np.float32)
        saved = np.memmap(run / "param_iter", dtype=np.float32, mode="r")
        count, remainder = divmod(saved.size, nvalues)
        if (remainder or count != int(iterations[-1]) + 1
                or final.size != nvalues or gfinal.size != nvalues
                or not np.array_equal(saved[-nvalues:], final)
                or not np.isfinite(gfinal).all()):
            raise ValueError("final model/gradient files do not match the completed history")
        costs = [float(v) for v in re.findall(
            r"^scaled fcost=([\deE.+-]+)", (run / "out.log").read_text(), re.M)]
        if len(costs) < 2 or not np.isclose(costs[0], data[0, 1], rtol=0.005):
            raise ValueError("out.log does not match the optimizer's initial objective")
        iterations = np.append(iterations, count)
        objective = np.append(objective, costs[-1] / costs[0])
        gradient = np.append(gradient, np.linalg.norm(gfinal.astype(float)) / data[0, 3])
    return iterations, objective, gradient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--run-label", default="l-BFGS")
    parser.add_argument("--compare-run-dir", type=Path,
                        help="second run to compare with the primary run")
    parser.add_argument("--compare-label", default="Gauss–Newton")
    parser.add_argument("--output", type=Path, default=RUN / "convergence.png",
                        help="output name; both .png and .pdf are written")
    args = parser.parse_args()
    histories = [(args.run_label, convergence_history(args.run_dir), "#0072B2", "o")]
    if args.compare_run_dir is not None:
        histories.append((args.compare_label, convergence_history(args.compare_run_dir),
                          "#D55E00", "s"))
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for column, (ax, title, label) in enumerate(zip(
            axes, ("(a) Objective", "(b) Gradient norm"),
            (r"$\Phi_k/\Phi_0$", r"$\|g_k\|_2/\|g_0\|_2$"))):
        for name, history, color, marker in histories:
            ax.semilogy(history[0], history[column + 1], color=color, linewidth=1.6,
                        marker=marker, markersize=3, markevery=5, label=name)
        ax.set(xlabel="Outer update" if args.compare_run_dir else "l-BFGS update",
               ylabel=label, xlim=(0, max(h[1][0][-1] for h in histories)))
        ax.set_title(title, loc="left")
        ax.grid(True, which="both", alpha=0.22)
        if args.compare_run_dir:
            ax.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        output = args.output.with_suffix(suffix)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Wrote {output}")
    plt.close(fig)
    for name, (iterations, objective, gradient), _, _ in histories:
        print(f"{name}, update {int(iterations[-1])}: "
              f"normalized objective = {objective[-1]:.8g}, "
              f"normalized gradient = {gradient[-1]:.8g}")


if __name__ == "__main__":
    main()
