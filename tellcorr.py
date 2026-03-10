#!/usr/bin/env python
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


ORDERS_PER_CHIP = {
    'L3244': 6, 'L3262': 6, 'L3302': 6, 'L3340': 6,
    'L3377': 7, 'L3412': 7, 'L3426': 7,
    'M4187': 6, 'M4211': 5, 'M4266': 7, 'M4318': 6,
    'M4368': 6, 'M4416': 6, 'M4461': 5, 'M4504': 5, 'M4519': 7,
}

OSET_OVERRIDE = {
    'M4211': '2:16',
}


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


def vipere_order_to_chip_order(vorder, max_order):
    """vipere order 1 = chip1 of highest DRS order number, etc.

    max_order is the highest order number in the FITS (e.g. 7 for M4504),
    matching vipere's own n_orders = max(order numbers).
    """
    order_idx, det0 = divmod(vorder - 1, 3)
    chip = det0 + 1
    order_drs = max_order - order_idx
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
        oset = OSET_OVERRIDE.get(setting, f"1:{n_orders * 3}")

    cmd = [
        'uvx', '--from', str(Path.home() / 'vipere.git'), 'vipere',
        str(extracted_fits),
        '-oset', oset,
        '-o', 'tellfit',
        '-plot', '0',
        '-vcut', '0',
    ]
    print(f"Running: {' '.join(cmd)}")
    env = os.environ | {'MPLBACKEND': 'Agg'}
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print("vipere stdout:", result.stdout[-2000:] if result.stdout else "")
        print("vipere stderr:", result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError(f"vipere failed with return code {result.returncode}")

    pardat = os.path.join(workdir, 'tellfit.par.dat')
    resdir = os.path.join(workdir, 'res')
    if not os.path.exists(pardat):
        raise RuntimeError(f"vipere did not produce {pardat}")
    return pardat, resdir


def process_one(input_fits, oset=None):
    """Process a single extractedA or B file. Returns (outpath, hdul_tellcorr)."""
    inpath = Path(input_fits).resolve()
    hdul = fits.open(inpath)
    header = hdul[0].header
    setting = detect_setting(header)
    n_opc = ORDERS_PER_CHIP.get(setting)
    if n_opc is None:
        raise ValueError(f"Unknown setting: {setting}")

    # max DRS order number from FITS columns (what vipere uses internally)
    max_order = max(
        int(c.split('_')[0])
        for c in hdul[1].columns.names if c.endswith('_SPEC')
    )
    print(f"Setting: {setting}, {n_opc} orders/chip, max order {max_order}")

    workdir = tempfile.mkdtemp(prefix='tellcorr_')
    try:
        link = os.path.join(workdir, inpath.name)
        os.symlink(inpath, link)

        pardat, resdir = run_vipere(link, workdir, setting, oset=oset)
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

        for ext_idx in [1, 2, 3]:
            chip = ext_idx
            table = hdul[ext_idx]
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

    # collect all order numbers
    orders = sorted(set(
        int(c.split('_')[0])
        for c in hdul_orig_a[1].columns.names if c.endswith('_SPEC')
    ))

    for odrs in orders:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 5), height_ratios=[3, 1],
            sharex=True, gridspec_kw={'hspace': 0.05}
        )

        wl_col = f"{odrs:02d}_01_WL"
        spec_col = f"{odrs:02d}_01_SPEC"
        tell_col = f"{odrs:02d}_01_TELLUR"
        cont_col = f"{odrs:02d}_01_CONT"

        for ext_idx in [1, 2, 3]:
            wl_a = hdul_orig_a[ext_idx].data[wl_col]
            spec_a = hdul_orig_a[ext_idx].data[spec_col]
            wl_b = hdul_orig_b[ext_idx].data[wl_col]
            spec_b = hdul_orig_b[ext_idx].data[spec_col]

            tell_a = hdul_tc_a[ext_idx].data[tell_col]
            tell_b = hdul_tc_b[ext_idx].data[tell_col]
            cont_a = hdul_tc_a[ext_idx].data[cont_col]
            cont_b = hdul_tc_b[ext_idx].data[cont_col]

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
                res_a = np.full_like(spec_a, np.nan)
                res_a[ok_a] = spec_a[ok_a] - model_a[ok_a]
                ax2.plot(wl_a, res_a, color='C0', lw=0.4, alpha=0.8)
            if ok_b.any():
                res_b = np.full_like(spec_b, np.nan)
                res_b[ok_b] = spec_b[ok_b] - model_b[ok_b]
                ax2.plot(wl_b, res_b, color='C1', lw=0.4, alpha=0.8)

        ax2.axhline(0, color='k', lw=0.5, alpha=0.5)

        ax1.plot([], [], color='C0', lw=1, label='A')
        ax1.plot([], [], color='C1', lw=1, label='B')
        ax1.plot([], [], color='k', lw=1, alpha=0.6, label='telluric model')
        ax1.legend(loc='upper right', fontsize=8)

        ax1.set_ylabel('flux [ADU]')
        ax2.set_ylabel('residual')
        ax2.set_xlabel('wavelength [nm]')
        ax1.set_title(f'{dirname}  order {odrs:02d}', fontsize=10)

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

    print(f"\n--- Processing B ---")
    tc_b_path = process_one(ext_b, oset=oset)

    print(f"\n--- Making plots ---")
    hdul_tc_a = fits.open(tc_a_path)
    hdul_tc_b = fits.open(tc_b_path)
    make_plots(dir_path, hdul_orig_a, hdul_tc_a, hdul_orig_b, hdul_tc_b)


def main():
    parser = argparse.ArgumentParser(
        description='Telluric-correct CRIRES+ extracted spectra using vipere')
    parser.add_argument('dir', help='Reduction directory containing '
                        'extractedA.fits and extractedB.fits')
    parser.add_argument('--oset', help='Override vipere -oset range (e.g. "1:15")')
    args = parser.parse_args()
    process_dir(args.dir, oset=args.oset)


if __name__ == '__main__':
    main()
