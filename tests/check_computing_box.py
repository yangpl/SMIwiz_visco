#!/usr/bin/env python3
"""Exercise generated TLM/adjoint operators with and without computing boxes.

Build with `make -C src SMIwiz`, then run `python3 tests/check_computing_box.py`.
Requires a working local MPI runtime; fixtures and logs remain under /tmp.
"""
from array import array
from pathlib import Path
import math
import os
import subprocess
import tempfile


def main():
    binary = Path(__file__).resolve().parents[1] / "bin/SMIwiz"
    root = Path(tempfile.mkdtemp(prefix="smiwiz-box-test-"))
    print(f"Test files: {root}", flush=True)
    for n3, order, freesurf, nrelax in [(1, 4, 1, 0), (1, 8, 0, 3),
                                      (13, 4, 0, 0), (13, 8, 1, 3)]:
        for ibox in (0, 1):
            name = f"n3_{n3}_order{order}_sls{nrelax}_ibox{ibox}"
            work = root / name
            work.mkdir()
            nt, n1, n2, dt = 180, 25, 29, 0.0005
            for field, value in [("vp", 1800), ("rho", 2000), ("qinv", .01)]:
                (work / field).write_bytes(array("f", [value] * (n1*n2*n3)).tobytes())
            wavelet = []
            for it in range(nt):
                a = (math.pi * 35 * (it*dt-.03))**2
                wavelet.append((1-2*a)*math.exp(-a))
            (work / "stf").write_bytes(array("f", wavelet).tobytes())
            y = 60 if n3 > 1 else 0
            geometry = ["z x y dip azimuth src/rec", f"51 140 {y} 0 0 0"]
            geometry += [f"50 {x} {y} 0 0 1" for x in (60, 100, 140, 180, 220)]
            (work / "acqui.txt").write_text("\n".join(geometry) + "\n")
            (work / "dat_0001").write_bytes(array("f", [0] * (5*nt)).tobytes())
            args = dict(mode=3, test_tlm_adjoint=1, acquifile="acqui.txt",
                        vpfile="vp", rhofile="rho", qinvfile="qinv", stffile="stf",
                        nt=nt, dt=dt, n1=n1, n2=n2, n3=n3, d1=10, d2=10, d3=10,
                        nb=8, nsnap=5, order=order, freesurf=freesurf,
                        nrelax=nrelax, ibox=ibox, family=1,
                        npar=3 if nrelax else 2, idxpar="1,2,3" if nrelax else "1,2")
            command = [str(binary)] + [f"{key}={value}" for key, value in args.items()]
            (work / "command.txt").write_text(" ".join(command) + "\n")
            with (work / "run.log").open("w") as log:
                result = subprocess.run(command, cwd=work, stdout=log,
                                        stderr=subprocess.STDOUT,
                                        env=dict(os.environ, OMP_NUM_THREADS="1"),
                                        timeout=180)
            output = (work / "run.log").read_text()
            assert result.returncode == 0, f"{name} failed; see {work / 'run.log'}"
            assert "adjoint test:" in output, f"{name}: consistency test did not run"
            print(name, flush=True)
            for line in output.splitlines():
                if "relative" in line:
                    print(f"  {line}", flush=True)
    print("All computing-box TLM/adjoint checks passed.", flush=True)


if __name__ == "__main__":
    main()
