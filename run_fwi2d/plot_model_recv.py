#!/usr/bin/env python3
"""Plot Marmousi Vp with receivers — model fills axis area exactly."""
import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, '..', 'Marmousi', 'vp_true.bin')
acqui_file = os.path.join(script_dir, 'acqui.txt')

nz, nx = 151, 461
dz = dx = 20.0
zmax = (nz - 1) * dz
xmax = (nx - 1) * dx

vp = np.fromfile(model_file, dtype=np.float32).reshape((nz, nx), order='F')

recv_x, recv_z = [], []
src_x, src_z = None, None
with open(acqui_file) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        zz, xx = float(parts[0]), float(parts[1])
        if int(parts[5]) == 0:
            src_z, src_x = zz, xx
        else:
            recv_z.append(zz)
            recv_x.append(xx)

fig, ax = plt.subplots(figsize=(10, 3.27))
ax.imshow(vp, cmap='viridis', aspect='auto',
          extent=[0, xmax, zmax, 0],
          vmin=1500, vmax=5500)

if recv_x:
    ax.scatter(recv_x, recv_z, s=2, c='white', marker='.', alpha=0.4,
               label=f'Recv ({len(recv_x)})')
if src_x is not None:
    ax.scatter(src_x, src_z, s=60, c='red', marker='*',
               edgecolors='white', linewidths=0.5,
               label='Source', zorder=5)

ax.set_xlim(0, xmax)
ax.set_ylim(zmax, 0)
ax.margins(0)
ax.set_xlabel('x (m)')
ax.set_ylabel('z (m)')
ax.legend(loc='lower right', markerscale=0.6, fontsize=7)

plt.colorbar(ax.images[0], ax=ax, label='Vp (m/s)', shrink=0.85, pad=0.02)

out = os.path.join(script_dir, 'model_with_receivers.png')
plt.savefig(out, dpi=150, bbox_inches='tight', pad_inches=0)
print(f'Saved: {out}')
plt.close()