#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["astropy", "numpy"]
# ///
"""Update Wavelength polynomials in _tw.fits from vipere telluric fits.

For orders with good vipere fits: use vipere wavelength solution.
For all others: restore from extracted spectrum WL column (original pipeline calibration).
"""

import numpy as np
from astropy.io import fits
from pathlib import Path

BASE = Path(__file__).parent
TW_DIR = BASE.parent
PRMS_MAX = 10
SHIFT_MAX_KMS = 10  # default max shift from pipeline
SHIFT_MAX_GOOD = 40  # allow larger shift if vipere fit is very good
PRMS_GOOD = 5  # prms threshold for "very good" fit


def vipere_order_to_crires(order_idx, hdu):
    det = ((order_idx - 1) % 3) + 1
    spec_order = ((order_idx - 1) // 3)
    ext = f"CHIP{det}.INT1"
    data = hdu[ext].data
    n_orders = max(int(c.split('_')[0]) for c in data.names if c.endswith('_SPEC'))
    odr = n_orders - spec_order
    return det, odr


def compute_xcen(hdu, chip, odr):
    ext = f'CHIP{chip}.INT1'
    data = hdu[ext].data
    col_spec = f'{odr:02d}_01_SPEC'
    col_wl = f'{odr:02d}_01_WL'
    if col_spec not in data.names:
        return None
    spec = data[col_spec]
    wl = data[col_wl]
    good = np.isfinite(spec) & (spec != 0) & np.isfinite(wl) & (wl > 0)
    if good.sum() < 50:
        return None
    return np.mean(np.where(good)[0]) + 18


def vipere_to_tw_poly(wave0, wave1, wave2, xcen):
    """Convert vipere polynomial (Angstrom, centered on xcen) to _tw.fits (nm, centered on 0)."""
    c0 = (wave0 - wave1 * xcen + wave2 * xcen**2) / 10
    c1 = (wave1 - 2 * wave2 * xcen) / 10
    c2 = wave2 / 10
    return c0, c1, c2


def wl_poly_from_extracted(hdu, chip, odr):
    """Fit quadratic to extracted WL column to recover original pipeline wavelength polynomial."""
    ext = f'CHIP{chip}.INT1'
    data = hdu[ext].data
    col_wl = f'{odr:02d}_01_WL'
    col_spec = f'{odr:02d}_01_SPEC'
    if col_wl not in data.names:
        return None
    wl = data[col_wl]
    spec = data[col_spec]
    good = np.isfinite(wl) & (wl > 0) & np.isfinite(spec) & (spec != 0)
    if good.sum() < 50:
        return None
    pix = np.arange(len(wl))
    c = np.polyfit(pix[good], wl[good], 2)
    return c[2], c[1], c[0]  # c0, c1, c2 in ascending order


settings = sorted(d.name for d in BASE.iterdir() if d.is_dir()
                  and (d / "telluricA.par.dat").exists()
                  and (d / "telluricA.par.dat").stat().st_size > 100)

for setting in settings:
    sdir = BASE / setting
    tw_path = TW_DIR / f'{setting}_tw.fits'
    if not tw_path.exists():
        print(f'{setting}: no _tw.fits, skipping')
        continue

    parA = np.genfromtxt(sdir / "telluricA.par.dat", names=True, dtype=None, encoding=None)
    if parA.ndim == 0:
        parA = parA.reshape(1)

    hduA = fits.open(sdir / "cr2res_obs_nodding_extractedA.fits")
    tw = fits.open(tw_path, mode='update')

    # build lookup: (chip, odr) -> vipere row
    vipere_lookup = {}
    for row_par in parA:
        oi = int(row_par['order'])
        chip, odr = vipere_order_to_crires(oi, hduA)
        vipere_lookup[(chip, odr)] = row_par

    print(f'{setting}:')
    for chip_idx in [1, 2, 3]:
        ext = f'CHIP{chip_idx}.INT1'
        tab = tw[ext].data

        for tw_row in tab:
            odr = tw_row['Order']

            # get pipeline wavelength from extracted spectrum
            pipeline = wl_poly_from_extracted(hduA, chip_idx, odr)
            if pipeline is None:
                continue
            c0_pipe, c1_pipe, c2_pipe = pipeline
            wl_pipe = c0_pipe + c1_pipe * 1024 + c2_pipe * 1024**2

            # try vipere
            key = (chip_idx, odr)
            use_vipere = False
            if key in vipere_lookup:
                row_par = vipere_lookup[key]
                prms = row_par['prms']
                if 0 < prms < PRMS_MAX:
                    xcen = compute_xcen(hduA, chip_idx, odr)
                    if xcen is not None:
                        c0_v, c1_v, c2_v = vipere_to_tw_poly(
                            row_par['wave0'], row_par['wave1'], row_par['wave2'], xcen)
                        wl_vip = c0_v + c1_v * 1024 + c2_v * 1024**2
                        shift_kms = abs(wl_vip - wl_pipe) / wl_pipe * 3e5
                        max_shift = SHIFT_MAX_GOOD if prms < PRMS_GOOD else SHIFT_MAX_KMS
                        if shift_kms < max_shift:
                            tw_row['Wavelength'] = [c0_v, c1_v, c2_v]
                            diff_kms = (wl_vip - wl_pipe) / wl_pipe * 3e5
                            print(f'  CHIP{chip_idx} O{odr}: vipere  {wl_pipe:.1f} -> {wl_vip:.1f} nm  ({diff_kms:+.1f} km/s)  prms={prms:.1f}')
                            use_vipere = True
                        else:
                            print(f'  CHIP{chip_idx} O{odr}: REJECT  shift={shift_kms:.0f} km/s  prms={prms:.1f}')

            if not use_vipere:
                tw_row['Wavelength'] = [c0_pipe, c1_pipe, c2_pipe]

    tw.flush()
    tw.close()
    hduA.close()

print('\nDone.')
