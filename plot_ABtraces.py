#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "astropy", "matplotlib"]
# ///
"""Plot combinedA image with A and B trace positions overlaid."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits


def get_extract_height(header):
    for i in range(1, 30):
        name_key = f'ESO PRO REC1 PARAM{i} NAME'
        val_key = f'ESO PRO REC1 PARAM{i} VALUE'
        if header.get(name_key) == 'extract_height':
            return int(header[val_key])
    return 45


def eval_trace(coeffs, nx=2048):
    x = np.arange(nx, dtype=float)
    return np.polynomial.polynomial.polyval(x, coeffs)


def plot_dir(dir_path):
    dir_path = Path(dir_path).resolve()

    comb_file = dir_path / 'cr2res_obs_nodding_combinedA.fits'
    tw_a_file = dir_path / 'cr2res_obs_nodding_trace_wave_A.fits'
    tw_b_file = dir_path / 'cr2res_obs_nodding_trace_wave_B.fits'

    if not comb_file.exists():
        print(f"No combinedA.fits in {dir_path.name}")
        return
    if not tw_a_file.exists() or not tw_b_file.exists():
        print(f"No trace_wave files in {dir_path.name}")
        return

    comb = fits.open(comb_file)
    tw_a = fits.open(tw_a_file)
    tw_b = fits.open(tw_b_file)

    height = get_extract_height(tw_a[0].header)
    half_h = height / 2.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(2048)

    for idx, chip in enumerate([1, 2, 3]):
        ax = axes[idx]
        extname = f'CHIP{chip}.INT1'

        img = np.abs(comb[extname].data.astype(float))
        vmax = np.nanpercentile(img, 95)
        ax.imshow(img, origin='lower', aspect='auto', cmap='plasma',
                  vmin=0, vmax=vmax)

        traces_a = tw_a[extname].data
        traces_b = tw_b[extname].data

        for traces in [traces_a, traces_b]:
            for row in traces:
                y = eval_trace(row['All'])
                for yy in [y + half_h, y - half_h]:
                    ax.plot(x, yy, color='black', lw=1.2, ls='-', alpha=1)
                    ax.plot(x, yy, color='white', lw=1.2, ls=(0, (4, 4)), alpha=1)

        ax.set_xlim(0, 2047)
        ax.set_ylim(0, 2047)
        ax.set_title(f'CHIP{chip}')
        ax.set_xlabel('pixel')
        if idx == 0:
            ax.set_ylabel('pixel')
        else:
            ax.set_yticklabels([])

    fig.suptitle(f'{dir_path.name}  (extract_height={height})', fontsize=10)
    fig.tight_layout()

    outpath = dir_path / 'ABtraces.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot combinedA with A/B trace positions')
    parser.add_argument('dir', help='Reduction directory')
    args = parser.parse_args()
    plot_dir(args.dir)


if __name__ == '__main__':
    main()
