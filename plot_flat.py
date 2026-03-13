#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "astropy", "matplotlib"]
# ///
"""Plot master flat with trace positions and blaze functions for a flat reduction dir."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits


def eval_trace(coeffs, nx=2048):
    x = np.arange(nx, dtype=float)
    return np.polynomial.polynomial.polyval(x, coeffs)


def plot_dir(dir_path):
    dir_path = Path(dir_path).resolve()

    flat_file = dir_path / 'cr2res_cal_flat_Open_master_flat.fits'
    tw_file = dir_path / 'cr2res_cal_flat_Open_tw.fits'
    blaze_file = dir_path / 'cr2res_cal_flat_Open_blaze.fits'

    if not flat_file.exists():
        print(f"No master_flat.fits in {dir_path.name}")
        return

    flat = fits.open(flat_file)
    tw = fits.open(tw_file) if tw_file.exists() else None
    blaze = fits.open(blaze_file) if blaze_file.exists() else None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             gridspec_kw={'height_ratios': [3, 1]})
    x = np.arange(2048)

    for idx, chip in enumerate([1, 2, 3]):
        ax_img = axes[0, idx]
        ax_blz = axes[1, idx]
        extname = f'CHIP{chip}.INT1'

        # top: master flat with traces
        img = flat[extname].data.astype(float)
        vmin = np.nanpercentile(img, 15)
        vmax = np.nanpercentile(img, 95)
        ax_img.imshow(img, origin='lower', aspect='auto', cmap='plasma',
                      vmin=vmin, vmax=vmax)

        if tw is not None:
            traces = tw[extname].data
            for row in traces:
                y_upper = eval_trace(row['Upper'])
                y_lower = eval_trace(row['Lower'])
                for yy in [y_upper, y_lower]:
                    ax_img.plot(x, yy, color='black', lw=1.2, ls='-', alpha=1)
                    ax_img.plot(x, yy, color='white', lw=1.2, ls=(0, (4, 4)), alpha=1)

        ax_img.set_xlim(0, 2047)
        ax_img.set_ylim(0, 2047)
        ax_img.set_title(f'CHIP{chip}')
        if idx == 0:
            ax_img.set_ylabel('pixel')
        else:
            ax_img.set_yticklabels([])

        # bottom: blaze functions
        if blaze is not None:
            cols = blaze[extname].columns.names
            for col in sorted(c for c in cols if c.endswith('_SPEC')):
                spec = blaze[extname].data[col]
                ok = np.isfinite(spec) & (spec != 0)
                if ok.any():
                    ax_blz.plot(x[ok], spec[ok], lw=0.8)

        ax_blz.set_xlim(0, 2047)
        ax_blz.set_xlabel('pixel')
        if idx == 0:
            ax_blz.set_ylabel('blaze')
        else:
            ax_blz.set_yticklabels([])

    fig.suptitle(dir_path.name, fontsize=10)
    fig.tight_layout()

    outpath = dir_path / 'flat_traces.png'
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Wrote {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot master flat with traces and blaze functions')
    parser.add_argument('dir', help='Flat reduction directory')
    args = parser.parse_args()
    plot_dir(args.dir)


if __name__ == '__main__':
    main()
