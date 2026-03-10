#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Fit a single straight line tilt(wavelength) for L and M bands separately."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent

with open(BASE / 'tilt_fits.json') as f:
    data = json.load(f)

measurements = data['measurements']

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
chip_colors = {1: 'C0', 2: 'C1', 3: 'C2'}
chip_markers = {1: 'o', 2: 's', 3: 'D'}

results = {}

for ax, band in zip(axes, ['L', 'M']):
    sub = [m for m in measurements if m['band'] == band]
    wl = np.array([m['wl_nm'] for m in sub])
    tilt = np.array([m['tilt'] for m in sub])
    chips = np.array([m['chip'] for m in sub])

    # iterative sigma clip + linear fit
    mask = np.ones(len(wl), dtype=bool)
    for iteration in range(5):
        c = np.polyfit(wl[mask], tilt[mask], 1)
        res = tilt - np.polyval(c, wl)
        sigma = np.std(res[mask])
        mask = np.abs(res) < 3 * sigma

    rms = np.std(res[mask])
    n_rej = (~mask).sum()

    for chip in [1, 2, 3]:
        sel = chips == chip
        ax.plot(wl[sel & mask], tilt[sel & mask],
                marker=chip_markers[chip], color=chip_colors[chip],
                ms=2, lw=0, alpha=0.3, label=f'CHIP{chip}')
        ax.plot(wl[sel & ~mask], tilt[sel & ~mask],
                marker=chip_markers[chip], color='gray',
                ms=2, lw=0, alpha=0.2)

    wl_plot = np.linspace(wl.min(), wl.max(), 100)
    ax.plot(wl_plot, np.polyval(c, wl_plot), 'k-', lw=2,
            label=f'slope={c[0]*1e3:.3f}e-3/nm, rms={rms:.4f}')
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.set_ylabel('tilt (dx/dy)')
    ax.set_title(f'{band} band: {mask.sum()} pts, {n_rej} rejected')
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.3)

    results[band] = {'slope': float(c[0]), 'intercept': float(c[1]),
                     'rms': float(rms), 'n_pts': int(mask.sum()),
                     'wl_min': float(wl[mask].min()), 'wl_max': float(wl[mask].max())}
    print(f'{band}: tilt = {c[0]:.6e} * wl_nm + {c[1]:.4f}  (rms={rms:.5f}, {mask.sum()} pts, {n_rej} rejected)')

axes[1].set_xlabel('Wavelength [nm]')
plt.tight_layout()
plt.savefig(BASE / 'tilt_linear_fit.png', dpi=150)
print(f'Saved tilt_linear_fit.png')

with open(BASE / 'tilt_linear_fit.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved tilt_linear_fit.json')
