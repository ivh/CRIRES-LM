#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "astropy", "matplotlib"]
# ///
"""Telluric-correct a CRIRES+ extractedA/B spectrum using vipere.

Takes a reduction directory as input, processes both extractedA and extractedB,
saves _tellcorr.fits files and a tellcorrAB.png diagnostic plot.
"""

import argparse
import json
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BASE = Path(__file__).parent

ORDERS_PER_CHIP = {
    'L3244': 6, 'L3262': 6, 'L3302': 6, 'L3340': 6,
    'L3377': 7, 'L3412': 7, 'L3426': 7,
    'M4187': 6, 'M4211': 6, 'M4266': 7, 'M4318': 6,
    'M4368': 6, 'M4416': 6, 'M4461': 6, 'M4504': 6, 'M4519': 7,
}

OSET_OVERRIDE = {}


def detect_setting(header):
    wlen_id = header.get('ESO INS WLEN ID', '')
    if wlen_id:
        return wlen_id
    cwlen = header.get('ESO INS WLEN CWLEN', 0)
    filt = header.get('ESO INS FILT1 NAME', '')
    band = 'L' if 'L' in filt or cwlen < 4200 else 'M'
    return f"{band}{int(cwlen)}"


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


def vipere_order_to_chip_order(vorder, max_per_chip):
    """vipere order 1 = chip1 of highest DRS order number, etc.

    max_per_chip is a dict {1: max1, 2: max2, 3: max3} giving the highest
    order number per chip in the extracted FITS.  Vipere determines its
    n_orders per chip independently, so chips with fewer orders get a
    different mapping.
    """
    order_idx, det0 = divmod(vorder - 1, 3)
    chip = det0 + 1
    order_drs = max_per_chip[chip] - order_idx
    return chip, order_drs


def reconstruct_model(spec, res_pixels, res_values, npix=2048):
    """model = observed - residual, interpolated to fill gaps."""
    pix_ok = res_pixels.astype(int)
    model_ok = spec[pix_ok] - res_values

    model = np.full(npix, np.nan)
    pmin, pmax = int(pix_ok[0]), int(pix_ok[-1])
    allpix = np.arange(pmin, pmax + 1)
    model[pmin:pmax + 1] = np.interp(allpix, pix_ok, model_ok)
    return model


def run_vipere(extracted_fits, workdir, setting, oset=None):
    n_orders = ORDERS_PER_CHIP.get(setting)
    if n_orders is None:
        raise ValueError(f"Unknown setting {setting}")

    if oset is None:
        oset = OSET_OVERRIDE.get(setting, f"1:{n_orders * 3 + 1}")

    env = os.environ | {'MPLBACKEND': 'Agg'}
    base_cmd = [
        'uv', 'run', '--with-editable', str(Path.home() / 'vipere.git'), 'vipere',
        str(extracted_fits),
        '-plot', '0',
        '-vcut', '100',
        '-deg_wave', '2',
        '-telluric', 'add',
        '-kapsig', '0',
    ]

    cmd = base_cmd + ['-oset', oset, '-deg_norm', '4', '-o', 'tellfit']
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print("vipere stdout:", result.stdout[-2000:] if result.stdout else "")
        print("vipere stderr:", result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError(f"vipere failed with return code {result.returncode}")

    pardat = os.path.join(workdir, 'tellfit.par.dat')
    resdir = os.path.join(workdir, 'res')
    if not os.path.exists(pardat):
        raise RuntimeError(f"vipere did not produce {pardat}")

    # retry failed orders with lower deg_norm
    missing = _find_missing_orders(pardat, oset)
    for vo in missing:
        retry_oset = f"{vo}:{vo + 1}"
        cmd2 = base_cmd + ['-oset', retry_oset, '-deg_norm', '2',
                           '-o', 'retry']
        print(f"  Retrying order {vo} with deg_norm=2")
        r2 = subprocess.run(cmd2, cwd=workdir, capture_output=True,
                            text=True, env=env)
        if r2.returncode != 0:
            continue
        retry_par = os.path.join(workdir, 'retry.par.dat')
        if not os.path.exists(retry_par):
            continue
        retry_rows = parse_pardat(retry_par)
        # read main header to match column layout
        with open(pardat) as f:
            main_header = f.readline().split()
        for row in retry_rows:
            vo_row = int(row.get('order', 0))
            if vo_row == 0 or row.get('prms', -1) < 0:
                continue
            # build line matching main header, padding missing columns
            vals = []
            for col in main_header:
                vals.append(str(row.get(col, 0)))
            with open(pardat, 'a') as f:
                f.write(' '.join(vals) + '\n')
            print(f"    Order {vo_row} recovered (prms={row.get('prms', -1):.1f}%)")

    return pardat, resdir


def _find_missing_orders(pardat, oset):
    """Find vipere orders in oset range but missing from par.dat."""
    pars = parse_pardat(pardat)
    fitted = set()
    for p in pars:
        o = int(p.get('order', 0))
        if o > 0 and p.get('prms', -1) >= 0:
            fitted.add(o)

    parts = oset.split(':')
    if len(parts) == 2:
        expected = set(range(int(parts[0]), int(parts[1])))
    else:
        return []
    return sorted(expected - fitted)


def process_one(input_fits, oset=None):
    """Process a single extractedA or B file. Returns (outpath, hdul_tellcorr)."""
    inpath = Path(input_fits).resolve()
    hdul = fits.open(inpath)
    if 'CHIP1.INT1' not in hdul or not hasattr(hdul['CHIP1.INT1'], 'columns'):
        print(f"  Skipping {inpath.name}: empty CHIP1.INT1 (bad esorex output?), deleting")
        hdul.close()
        inpath.unlink()
        return None
    header = hdul[0].header
    setting = detect_setting(header)
    n_opc = ORDERS_PER_CHIP.get(setting)
    if n_opc is None:
        raise ValueError(f"Unknown setting: {setting}")

    # per-chip max DRS order number from FITS columns (vipere determines
    # n_orders per chip independently)
    max_order = {}
    for chip in [1, 2, 3]:
        ext = f'CHIP{chip}.INT1'
        if ext in hdul:
            max_order[chip] = max(
                int(c.split('_')[0])
                for c in hdul[ext].columns.names if c.endswith('_SPEC')
            )
    print(f"Setting: {setting}, {n_opc} orders/chip, max order {max_order}")

    workdir = tempfile.mkdtemp(prefix='tellcorr_')
    try:
        # copy and sanitize before passing to vipere
        sanitized = os.path.join(workdir, inpath.name)
        san_hdul = fits.open(inpath)
        edge_mask = 5  # NaN out first/last N pixels per detector segment
        for chip in [1, 2, 3]:
            extname = f'CHIP{chip}.INT1'
            if extname not in san_hdul:
                continue
            for col in san_hdul[extname].columns.names:
                if col.endswith('_SPEC'):
                    spec = san_hdul[extname].data[col]
                    # mask detector edge pixels (extraction artifacts)
                    spec[:edge_mask] = np.nan
                    spec[-edge_mask:] = np.nan
                    # mask extreme negative spikes (bad nodding subtraction)
                    med = np.nanmedian(spec)
                    bad = spec < -abs(med)
                    if bad.any():
                        spec[bad] = np.nan
                        print(f"  Sanitized {bad.sum()} extreme negative pixels "
                              f"in {extname} {col}")
        san_hdul.writeto(sanitized, overwrite=True)
        san_hdul.close()

        pardat, resdir = run_vipere(sanitized, workdir, setting, oset=oset)
        pars = parse_pardat(pardat)
        if not pars:
            raise RuntimeError("vipere par.dat is empty")
        print(f"  {len(pars)} order fits in par.dat")

        fits_data = {}
        for par in pars:
            vorder = int(par.get('order', 0))
            if vorder == 0:
                continue

            chip, order_drs = vipere_order_to_chip_order(vorder, max_order)

            resfile = os.path.join(resdir, f"000_{vorder:03d}.dat")
            if not os.path.exists(resfile):
                print(f"  Warning: no residual for vipere order {vorder} "
                      f"(chip{chip} order{order_drs:02d})")
                continue

            res_data = np.loadtxt(resfile)
            if res_data.ndim != 2 or res_data.shape[1] != 2:
                continue

            prms = par.get('prms', -1)
            if prms < 0:
                print(f"  Skipping chip{chip} order{order_drs:02d}: "
                      f"failed fit (prms={prms:.1f})")
                continue

            xcen = float(np.nanmean(res_data[:, 0])) + 18
            fits_data[(chip, order_drs)] = {
                'par': par,
                'res_pix': res_data[:, 0],
                'res_val': res_data[:, 1],
                'xcen': xcen,
            }

        out_hdul = fits.HDUList([hdul[0].copy()])

        for chip in [1, 2, 3]:
            table = hdul[f'CHIP{chip}.INT1']
            npix = table.data.shape[0]

            orders_in_chip = sorted(set(
                int(c.split('_')[0])
                for c in table.columns.names if c.endswith('_SPEC')
            ))

            col_arrays = {}
            for col in table.columns:
                col_arrays[col.name] = table.data[col.name].copy()

            for odrs in orders_in_chip:
                spec_col = f"{odrs:02d}_01_SPEC"
                spec = col_arrays[spec_col]
                tellcol = f"{odrs:02d}_01_TELLUR"
                contcol = f"{odrs:02d}_01_CONT"

                key = (chip, odrs)
                if key in fits_data:
                    d = fits_data[key]
                    par = d['par']
                    model = reconstruct_model(spec, d['res_pix'], d['res_val'], npix)
                    prms = par.get('prms', -1)

                    # continuum polynomial from par.dat
                    xcen = d['xcen']
                    norm_coeffs = []
                    for i in range(10):
                        k = f'norm{i}'
                        if k not in par:
                            break
                        norm_coeffs.append(par[k])
                    pixels = np.arange(npix)
                    continuum = np.polynomial.polynomial.polyval(
                        pixels - xcen, norm_coeffs)

                    # telluric = model / continuum (transmission 0-1)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        telluric = np.where(
                            np.isfinite(model) & (continuum > 0),
                            model / continuum,
                            np.nan
                        )

                    # corrected = observed / telluric
                    with np.errstate(divide='ignore', invalid='ignore'):
                        corrected = np.where(
                            np.isfinite(telluric) & (telluric > 0.01),
                            spec / telluric,
                            np.nan
                        )
                    print(f"  chip{chip} order{odrs:02d}: corrected (prms={prms:.1f}%)")
                else:
                    corrected = spec.copy()
                    telluric = np.full(npix, np.nan)
                    continuum = np.full(npix, np.nan)
                    print(f"  chip{chip} order{odrs:02d}: no fit, keeping original")

                col_arrays[spec_col] = corrected
                col_arrays[tellcol] = telluric
                col_arrays[contcol] = continuum

            coldefs = []
            for col in table.columns:
                coldefs.append(fits.Column(
                    name=col.name, format=col.format,
                    array=col_arrays[col.name]
                ))
            for odrs in orders_in_chip:
                tellcol = f"{odrs:02d}_01_TELLUR"
                coldefs.append(fits.Column(
                    name=tellcol, format='1D',
                    array=col_arrays[tellcol]
                ))
                contcol = f"{odrs:02d}_01_CONT"
                coldefs.append(fits.Column(
                    name=contcol, format='1D',
                    array=col_arrays[contcol]
                ))

            new_table = fits.BinTableHDU.from_columns(
                fits.ColDefs(coldefs), header=table.header, nrows=npix
            )
            new_table.name = table.name
            out_hdul.append(new_table)

        stem = inpath.stem
        outpath = inpath.parent / f"{stem}_tellcorr.fits"
        out_hdul.writeto(outpath, overwrite=True)
        print(f"  Wrote {outpath}")

        # save par.dat and xcen as sidecar for wavecorr.py
        ab = 'A' if 'extractedA' in stem else 'B'
        pardat_out = inpath.parent / f"tellfit_{ab}.par.dat"
        shutil.copy2(pardat, pardat_out)

        xcen_info = {f"{chip}_{odrs:02d}": d['xcen']
                     for (chip, odrs), d in fits_data.items()}
        xcen_out = inpath.parent / f"tellfit_{ab}_xcen.json"
        with open(xcen_out, 'w') as f:
            json.dump(xcen_info, f)

    finally:
        if os.environ.get('TELLCORR_KEEP_WORKDIR'):
            print(f"  Workdir kept: {workdir}")
        else:
            shutil.rmtree(workdir)

    return outpath


def make_plots(dir_path, hdul_orig_a, hdul_tc_a, hdul_orig_b, hdul_tc_b):
    """Make one plot per spectral order, each showing 3 chips side by side."""
    dirname = Path(dir_path).name

    # collect all order numbers across all chips
    orders = set()
    for chip in [1, 2, 3]:
        for c in hdul_orig_a[f'CHIP{chip}.INT1'].columns.names:
            if c.endswith('_SPEC'):
                orders.add(int(c.split('_')[0]))
    orders = sorted(orders)

    for odrs in orders:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 5), height_ratios=[3, 1],
            sharex=True, gridspec_kw={'hspace': 0.05}
        )

        wl_col = f"{odrs:02d}_01_WL"
        spec_col = f"{odrs:02d}_01_SPEC"
        tell_col = f"{odrs:02d}_01_TELLUR"
        cont_col = f"{odrs:02d}_01_CONT"

        for chip in [1, 2, 3]:
            extname = f'CHIP{chip}.INT1'
            if spec_col not in hdul_orig_a[extname].columns.names:
                continue
            wl_a = hdul_orig_a[extname].data[wl_col]
            spec_a = hdul_orig_a[extname].data[spec_col]
            wl_b = hdul_orig_b[extname].data[wl_col]
            spec_b = hdul_orig_b[extname].data[spec_col]

            tell_a = hdul_tc_a[extname].data[tell_col]
            tell_b = hdul_tc_b[extname].data[tell_col]
            cont_a = hdul_tc_a[extname].data[cont_col]
            cont_b = hdul_tc_b[extname].data[cont_col]

            # model = continuum * telluric transmission
            model_a = cont_a * tell_a
            model_b = cont_b * tell_b

            ok_a = np.isfinite(spec_a) & np.isfinite(model_a)
            ok_b = np.isfinite(spec_b) & np.isfinite(model_b)

            ax1.plot(wl_a, spec_a, color='C0', lw=0.5, alpha=0.8)
            ax1.plot(wl_b, spec_b, color='C1', lw=0.5, alpha=0.8)

            if ok_a.any():
                ax1.plot(wl_a, model_a, color='k', lw=0.7, alpha=0.6)
            if ok_b.any():
                ax1.plot(wl_b, model_b, color='k', lw=0.7, alpha=0.6)

            if ok_a.any():
                norm_a = np.full_like(spec_a, np.nan)
                norm_a[ok_a] = spec_a[ok_a] / model_a[ok_a]
                ax2.plot(wl_a, norm_a, color='C0', lw=0.4, alpha=0.8)
            if ok_b.any():
                norm_b = np.full_like(spec_b, np.nan)
                norm_b[ok_b] = spec_b[ok_b] / model_b[ok_b]
                ax2.plot(wl_b, norm_b, color='C1', lw=0.4, alpha=0.8)

        ax2.axhline(1, color='k', lw=0.5, alpha=0.5)

        ax1.plot([], [], color='C0', lw=1, label='A')
        ax1.plot([], [], color='C1', lw=1, label='B')
        ax1.plot([], [], color='k', lw=1, alpha=0.6, label='telluric model')
        ax1.legend(loc='upper right', fontsize=8)

        ax1.set_ylabel('flux [ADU]')
        ax2.set_ylabel('normalized')
        ax2.set_xlabel('wavelength [nm]')
        ax1.set_title(f'{dirname}  order {odrs:02d}', fontsize=10)

        # robust ylim: trim edge 10 pixels per chip, use 1-99 percentile
        for ax in (ax1, ax2):
            ydata = []
            for line in ax.get_lines():
                y = line.get_ydata()
                if len(y) > 20:
                    ydata.append(y[10:-10])
            if ydata:
                all_y = np.concatenate(ydata)
                all_y = all_y[np.isfinite(all_y)]
                if len(all_y) > 0:
                    lo, hi = np.percentile(all_y, [1, 99])
                    margin = 0.05 * (hi - lo) if hi > lo else 1
                    ax.set_ylim(lo - margin, hi + margin)

        for ax in (ax1, ax2):
            ax.tick_params(labelsize=8)

        outpng = Path(dir_path) / f'tellcorrAB_{odrs:02d}.png'
        fig.savefig(outpng, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Wrote {outpng}")


def process_dir(dir_path, oset=None):
    dir_path = Path(dir_path).resolve()
    ext_a = dir_path / 'cr2res_obs_nodding_extractedA.fits'
    ext_b = dir_path / 'cr2res_obs_nodding_extractedB.fits'

    if not ext_a.exists() or not ext_b.exists():
        raise FileNotFoundError(
            f"Need both extractedA and extractedB in {dir_path}")

    hdul_orig_a = fits.open(ext_a)
    hdul_orig_b = fits.open(ext_b)

    print(f"\n--- Processing A ---")
    tc_a_path = process_one(ext_a, oset=oset)
    if tc_a_path is None:
        return

    print(f"\n--- Processing B ---")
    tc_b_path = process_one(ext_b, oset=oset)
    if tc_b_path is None:
        return

    print(f"\n--- Making plots ---")
    hdul_tc_a = fits.open(tc_a_path)
    hdul_tc_b = fits.open(tc_b_path)
    make_plots(dir_path, hdul_orig_a, hdul_tc_a, hdul_orig_b, hdul_tc_b)


def replot(dir_path):
    dir_path = Path(dir_path).resolve()
    ext_a = dir_path / 'cr2res_obs_nodding_extractedA.fits'
    ext_b = dir_path / 'cr2res_obs_nodding_extractedB.fits'
    tc_a = dir_path / 'cr2res_obs_nodding_extractedA_tellcorr.fits'
    tc_b = dir_path / 'cr2res_obs_nodding_extractedB_tellcorr.fits'
    for f in [ext_a, ext_b, tc_a, tc_b]:
        if not f.exists():
            raise FileNotFoundError(f"Missing {f}")
    make_plots(dir_path, fits.open(ext_a), fits.open(tc_a),
               fits.open(ext_b), fits.open(tc_b))


def main():
    parser = argparse.ArgumentParser(
        description='Telluric-correct CRIRES+ extracted spectra using vipere')
    parser.add_argument('dir', help='Reduction directory containing '
                        'extractedA.fits and extractedB.fits')
    parser.add_argument('--oset', help='Override vipere -oset range (e.g. "1:15")')
    parser.add_argument('--plot-only', action='store_true',
                        help='Re-generate plots from existing _tellcorr.fits')
    args = parser.parse_args()
    if args.plot_only:
        replot(args.dir)
    else:
        process_dir(args.dir, oset=args.oset)


if __name__ == '__main__':
    main()
