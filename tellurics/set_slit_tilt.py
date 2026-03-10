#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["astropy", "numpy"]
# ///
"""Set SlitPolyB in _tw.fits files using linear tilt(wavelength) fits."""

import json
import numpy as np
from astropy.io import fits
from pathlib import Path

BASE = Path(__file__).parent

with open(BASE / 'tilt_linear_fit.json') as f:
    tilt_fit = json.load(f)

TW_DIR = BASE.parent
tw_files = sorted(TW_DIR.glob('*_tw.fits'))

for tw_path in tw_files:
    setting = tw_path.stem.replace('_tw', '')
    if setting.startswith('cr2res'):
        continue

    band = 'L' if setting.startswith('L') else 'M'
    fit = tilt_fit[band]
    slope, intercept = fit['slope'], fit['intercept']

    hdu = fits.open(tw_path, mode='update')
    print(f'{setting} ({band}):')

    for chip in [1, 2, 3]:
        ext = f'CHIP{chip}.INT1'
        tab = hdu[ext].data
        for row in tab:
            odr = row['Order']
            wl_poly = row['Wavelength']
            wl_center = wl_poly[0] + wl_poly[1] * 1024 + wl_poly[2] * 1024**2
            tilt = slope * wl_center + intercept
            row['SlitPolyB'] = [tilt, 0.0, 0.0]
            print(f'  CHIP{chip} O{odr}: wl={wl_center:.1f} nm -> tilt={tilt:.5f}')

    hdu.flush()
    hdu.close()

print('\nDone.')
