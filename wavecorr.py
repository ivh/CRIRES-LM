#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "astropy", "matplotlib"]
# ///
"""Update wavelength scale in _tellcorr.fits using vipere solutions.

For orders with a vipere fit: uses the fitted wavelength polynomial directly.
For orders without: fits a velocity correction dv(wavelength) across all 3 chips
and applies it to the pipeline wavelength. The correction is smooth because the
chips are at fixed relative positions.

Requires tellfit_{A,B}.par.dat and tellfit_{A,B}_xcen.json from tellcorr.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_pardat(parfile):
    with open(parfile) as f:
        header = f.readline().split()
        rows = []
        for line in f:
            vals = line.split()
            row = {}
            for k, v in zip(header, vals):
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = np.nan
            rows.append(row)
    return rows


def get_coeffs(par, prefix):
    coeffs = []
    for i in range(10):
        k = f'{prefix}{i}'
        if k not in par:
            break
        coeffs.append(par[k])
    return coeffs


def vipere_order_to_chip_order(vorder, max_per_chip):
    order_idx, det0 = divmod(vorder - 1, 3)
    chip = det0 + 1
    order_drs = max_per_chip[chip] - order_idx
    return chip, order_drs


MAX_DV_KMS = 100.0  # reject corrections larger than this
MAX_PRMS_FOR_CORRECTION = 30.0  # only use orders with prms < this for the dv fit


def plot_velocity_correction(vipere_wl, pipeline_wl, vipere_prms, dv_coeffs,
                             all_unfitted, outpath, ab=''):
    """Plot dv vs wavelength: fitted orders as dots, correction curve, unfitted markers."""
    chip_colors = {1: 'C0', 2: 'C1', 3: 'C2'}
    fig, ax = plt.subplots(figsize=(12, 5))

    for (chip, odrs), vip_wl in sorted(vipere_wl.items()):
        key = (chip, odrs)
        if key not in pipeline_wl:
            continue
        pipe_wl = pipeline_wl[key]
        idx = np.linspace(100, 1947, 50).astype(int)
        dv = 3e5 * (vip_wl[idx] - pipe_wl[idx]) / pipe_wl[idx]
        good = vipere_prms.get(key, 999) < MAX_PRMS_FOR_CORRECTION
        ax.scatter(pipe_wl[idx], dv, s=4, color=chip_colors[chip],
                   alpha=0.6 if good else 0.15, marker='o' if good else 'x')

    wl_range = np.linspace(
        min(w.min() for w in pipeline_wl.values()),
        max(w.max() for w in pipeline_wl.values()),
        500)

    if dv_coeffs is not None:
        dv_fit = np.polyval(dv_coeffs, wl_range)
        ax.plot(wl_range, dv_fit, 'k-', lw=1.5, label=f'fit (deg {len(dv_coeffs)-1})')
        ax.axhspan(-MAX_DV_KMS, MAX_DV_KMS, alpha=0.05, color='green')

    for chip, odrs in all_unfitted:
        key = (chip, odrs)
        if key not in pipeline_wl:
            continue
        wl_mid = pipeline_wl[key][1024]
        ax.axvline(wl_mid, color=chip_colors[chip], ls=':', alpha=0.4, lw=0.8)
        ax.text(wl_mid, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 5,
                f'{odrs}', fontsize=7, ha='center', va='bottom',
                color=chip_colors[chip])

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=chip_colors[i], marker='o', ls='',
                       markersize=4, label=f'CHIP{i}') for i in [1, 2, 3]]
    if dv_coeffs is not None:
        handles.append(Line2D([], [], color='k', lw=1.5, label='fit'))
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('dv (vipere - pipeline) [km/s]')
    ax.set_title(f'Velocity correction ({ab})' if ab else 'Velocity correction')
    ax.axhline(0, color='gray', lw=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Wrote {outpath}")


def read_pipeline_wl_from_tw(tw_path, hdul):
    """Read pipeline wavelengths from _tw.fits polynomials for all extracted orders."""
    tw = fits.open(tw_path)
    pixels = np.arange(2048, dtype=float)
    pipeline_wl = {}
    for chip in [1, 2, 3]:
        ext = f'CHIP{chip}.INT1'
        # which orders exist in the extracted spectrum
        extracted_orders = set(
            int(c.split('_')[0])
            for c in hdul[ext].columns.names if c.endswith('_WL')
        )
        for row in tw[ext].data:
            order = row['Order']
            if order in extracted_orders:
                wl = np.polyval(row['Wavelength'][::-1], pixels)
                pipeline_wl[(chip, order)] = wl
    tw.close()
    return pipeline_wl


def process_one(tellcorr_fits, pardat_file, xcen_file, ab='A'):
    """Update WL columns in a _tellcorr.fits."""
    hdul = fits.open(tellcorr_fits, mode='update')
    pars = parse_pardat(pardat_file)
    with open(xcen_file) as f:
        xcen_map = json.load(f)

    max_order = {}
    for chip in [1, 2, 3]:
        ext = f'CHIP{chip}.INT1'
        if ext in hdul:
            max_order[chip] = max(
                int(c.split('_')[0])
                for c in hdul[ext].columns.names if c.endswith('_SPEC')
            )

    chip_solutions = {1: [], 2: [], 3: []}

    for par in pars:
        vorder = int(par.get('order', 0))
        if vorder == 0:
            continue
        prms = par.get('prms', -1)
        if prms < 0:
            continue

        chip, odrs = vipere_order_to_chip_order(vorder, max_order)
        wcoeffs = get_coeffs(par, 'wave')
        if not wcoeffs:
            continue

        xcen_key = f"{chip}_{odrs:02d}"
        if xcen_key not in xcen_map:
            continue
        xcen = xcen_map[xcen_key]

        chip_solutions[chip].append((odrs, xcen, wcoeffs, prms))

    pixels = np.arange(2048, dtype=float)

    # read pipeline WL from _tw.fits (not from tellcorr, which may be stale)
    dir_path = Path(tellcorr_fits).parent
    tw_files = list(dir_path.glob('*_tw.fits'))
    if not tw_files:
        print("  No _tw.fits found, cannot determine pipeline WL")
        hdul.close()
        return
    pipeline_wl = read_pipeline_wl_from_tw(tw_files[0], hdul)

    # compute vipere WL for all fitted orders
    vipere_wl = {}
    vipere_prms = {}
    for chip, solutions in chip_solutions.items():
        for odrs, xcen, wcoeffs, prms in solutions:
            wl = np.polynomial.polynomial.polyval(
                pixels - xcen, wcoeffs) / 10.0
            vipere_wl[(chip, odrs)] = wl
            vipere_prms[(chip, odrs)] = prms

    # apply direct vipere WL for fitted orders
    for (chip, odrs), wl in vipere_wl.items():
        wl_col = f"{odrs:02d}_01_WL"
        ext = f'CHIP{chip}.INT1'
        if wl_col in hdul[ext].columns.names:
            hdul[ext].data[wl_col] = wl

    # identify unfitted orders across all chips
    all_unfitted = []
    for chip in [1, 2, 3]:
        ext = f'CHIP{chip}.INT1'
        orders_in_chip = sorted(set(
            int(c.split('_')[0])
            for c in hdul[ext].columns.names if c.endswith('_SPEC')
        ))
        fitted_set = set(s[0] for s in chip_solutions[chip])
        for odrs in orders_in_chip:
            if odrs not in fitted_set:
                all_unfitted.append((chip, odrs))

    n_fitted = len(vipere_wl)
    good_for_corr = {k for k, p in vipere_prms.items()
                     if p < MAX_PRMS_FOR_CORRECTION}
    n_good = len(good_for_corr)
    print(f"  {n_fitted} fitted orders ({n_good} with prms<{MAX_PRMS_FOR_CORRECTION}), "
          f"{len(all_unfitted)} unfitted")
    for chip in [1, 2, 3]:
        fitted = sorted(s[0] for s in chip_solutions[chip])
        unfitted = [o for c, o in all_unfitted if c == chip]
        print(f"    chip{chip}: fitted={fitted}, unfitted={unfitted}")

    dv_coeffs = None

    if all_unfitted and n_good >= 3:
        # collect (wavelength, dv) samples from good orders across all chips
        wl_samples = []
        dv_samples = []
        for (chip, odrs), vip_wl in vipere_wl.items():
            if (chip, odrs) not in good_for_corr:
                continue
            key = (chip, odrs)
            if key not in pipeline_wl:
                continue
            pipe_wl = pipeline_wl[key]
            idx = np.linspace(100, 1947, 20).astype(int)
            dv = 3e5 * (vip_wl[idx] - pipe_wl[idx]) / pipe_wl[idx]
            wl_samples.extend(pipe_wl[idx])
            dv_samples.extend(dv)

        wl_samples = np.array(wl_samples)
        dv_samples = np.array(dv_samples)

        deg = 1
        dv_coeffs = np.polyfit(wl_samples, dv_samples, deg)
        resid = dv_samples - np.polyval(dv_coeffs, wl_samples)
        print(f"  dv correction fit: {n_good} good orders, deg={deg}, "
              f"rms={np.std(resid)*1e3:.0f} m/s")

        for chip, odrs in all_unfitted:
            key = (chip, odrs)
            if key not in pipeline_wl:
                continue
            pipe_wl = pipeline_wl[key]
            dv_corr = np.polyval(dv_coeffs, pipe_wl)
            if np.max(np.abs(dv_corr)) > MAX_DV_KMS:
                print(f"    chip{chip} order {odrs:02d}: correction "
                      f"{np.median(dv_corr):.1f} km/s exceeds limit, "
                      f"keeping pipeline WL")
                continue
            corrected = pipe_wl * (1 + dv_corr / 3e5)
            wl_col = f"{odrs:02d}_01_WL"
            ext = f'CHIP{chip}.INT1'
            if wl_col in hdul[ext].columns.names:
                hdul[ext].data[wl_col] = corrected
                print(f"    chip{chip} order {odrs:02d}: corrected by "
                      f"{np.median(dv_corr):.2f} km/s")
    elif all_unfitted:
        print(f"  only {n_good} good orders (need 3), unfitted keep pipeline WL")

    plot_velocity_correction(
        vipere_wl, pipeline_wl, vipere_prms, dv_coeffs, all_unfitted,
        Path(tellcorr_fits).parent / f'wavecorr_{ab}.png', ab=ab)

    hdul.flush()
    hdul.close()


def process_dir(dir_path):
    dir_path = Path(dir_path).resolve()

    for old_png in dir_path.glob('wavecorr_*.png'):
        old_png.unlink()

    for ab in ['A', 'B']:
        tellcorr = dir_path / f'cr2res_obs_nodding_extracted{ab}_tellcorr.fits'
        pardat = dir_path / f'tellfit_{ab}.par.dat'
        xcen = dir_path / f'tellfit_{ab}_xcen.json'

        if not tellcorr.exists():
            print(f"  Skipping {ab}: no tellcorr file")
            continue
        if not pardat.exists() or not xcen.exists():
            print(f"  Skipping {ab}: no par.dat/xcen (run tellcorr.py first)")
            continue

        print(f"\n--- Wavelength correction {ab} ---")
        process_one(tellcorr, pardat, xcen, ab=ab)
        print(f"  Updated {tellcorr}")


def main():
    parser = argparse.ArgumentParser(
        description='Update wavelength scale in tellcorr FITS using '
                    'vipere solutions + velocity correction')
    parser.add_argument('dir', help='Reduction directory')
    args = parser.parse_args()
    process_dir(args.dir)


if __name__ == '__main__':
    main()
