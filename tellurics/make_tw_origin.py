#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["astropy", "numpy"]
# ///
"""Generate tw_origin.md: provenance table for each trace in every _tw.fits."""

import json
import numpy as np
from astropy.io import fits
from pathlib import Path

BASE = Path(__file__).parent
TW_DIR = BASE.parent
PRMS_MAX = 10
SHIFT_MAX_KMS = 10
SHIFT_MAX_GOOD = 40
PRMS_GOOD = 5


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
    c0 = (wave0 - wave1 * xcen + wave2 * xcen**2) / 10
    c1 = (wave1 - 2 * wave2 * xcen) / 10
    c2 = wave2 / 10
    return c0, c1, c2


# load tilt measurements to find wavelength coverage
with open(BASE / 'tilt_fits.json') as f:
    tilt_data = json.load(f)
meas = tilt_data['measurements']
# per-band wavelength ranges with actual vipere tilt data
tilt_wl = {}
for band in ['L', 'M']:
    wls = sorted(m['wl_nm'] for m in meas if m['band'] == band)
    tilt_wl[band] = (min(wls), max(wls))

# find gaps: wavelength ranges within band coverage that lack measurements
# bin measurements by 50nm to find gaps
tilt_gaps = {}
for band in ['L', 'M']:
    wls = np.array([m['wl_nm'] for m in meas if m['band'] == band])
    wl_min, wl_max = wls.min(), wls.max()
    bins = np.arange(wl_min, wl_max + 50, 50)
    counts, _ = np.histogram(wls, bins)
    gap_ranges = []
    for i, c in enumerate(counts):
        if c < 3:
            gap_ranges.append((bins[i], bins[i+1]))
    # merge adjacent gaps
    merged = []
    for g in gap_ranges:
        if merged and g[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], g[1])
        else:
            merged.append(g)
    tilt_gaps[band] = merged


def tilt_origin(band, wl_nm):
    """Determine if a trace's tilt comes from a region with vipere data or is interpolated."""
    wl_range = tilt_wl.get(band)
    if wl_range is None:
        return 'interp'
    if wl_nm < wl_range[0] or wl_nm > wl_range[1]:
        return 'extrap'
    for gap_lo, gap_hi in tilt_gaps.get(band, []):
        if gap_lo <= wl_nm <= gap_hi:
            return 'interp'
    return 'vipere'


settings = sorted(d.name for d in BASE.iterdir() if d.is_dir()
                  and (d / "telluricA.par.dat").exists()
                  and (d / "telluricA.par.dat").stat().st_size > 100)

# also include settings without vipere (L3340, M4519)
all_tw = sorted(TW_DIR.glob('*_tw.fits'))
all_settings = []
for tw_path in all_tw:
    s = tw_path.stem.replace('_tw', '')
    if s.startswith('cr2res'):
        continue
    all_settings.append(s)

rows = []

for setting in all_settings:
    tw_path = TW_DIR / f'{setting}_tw.fits'
    sdir = BASE / setting
    band = 'L' if setting.startswith('L') else 'M'

    has_vipere = (sdir / "telluricA.par.dat").exists() and (sdir / "telluricA.par.dat").stat().st_size > 100

    # build vipere lookup
    vipere_good = set()
    if has_vipere:
        parA = np.genfromtxt(sdir / "telluricA.par.dat", names=True, dtype=None, encoding=None)
        if parA.ndim == 0:
            parA = parA.reshape(1)
        hduA = fits.open(sdir / "cr2res_obs_nodding_extractedA.fits")

        for row_par in parA:
            oi = int(row_par['order'])
            prms = row_par['prms']
            if not (0 < prms < PRMS_MAX):
                continue
            chip, odr = vipere_order_to_crires(oi, hduA)
            xcen = compute_xcen(hduA, chip, odr)
            if xcen is None:
                continue
            c0, c1, c2 = vipere_to_tw_poly(row_par['wave0'], row_par['wave1'], row_par['wave2'], xcen)
            # get pipeline wl for comparison
            ext = f'CHIP{chip}.INT1'
            data = hduA[ext].data
            col_wl = f'{odr:02d}_01_WL'
            if col_wl not in data.names:
                continue
            wl_arr = data[col_wl]
            spec = data[f'{odr:02d}_01_SPEC']
            good = np.isfinite(wl_arr) & (wl_arr > 0) & np.isfinite(spec) & (spec != 0)
            if good.sum() < 50:
                continue
            pix = np.arange(len(wl_arr))
            cp = np.polyfit(pix[good], wl_arr[good], 2)
            wl_pipe = cp[2] + cp[1] * 1024 + cp[0] * 1024**2
            wl_vip = c0 + c1 * 1024 + c2 * 1024**2
            shift_kms = abs(wl_vip - wl_pipe) / wl_pipe * 3e5
            max_shift = SHIFT_MAX_GOOD if prms < PRMS_GOOD else SHIFT_MAX_KMS
            if shift_kms < max_shift:
                vipere_good.add((chip, odr))
        hduA.close()

    tw = fits.open(tw_path)
    for chip in [1, 2, 3]:
        ext = f'CHIP{chip}.INT1'
        tab = tw[ext].data
        for tw_row in tab:
            odr = tw_row['Order']
            wl_poly = tw_row['Wavelength']
            wl_center = wl_poly[0] + wl_poly[1] * 1024 + wl_poly[2] * 1024**2
            sb = tw_row['SlitPolyB'][0]

            wl_src = 'vipere' if (chip, odr) in vipere_good else 'pipeline'
            tilt_src = tilt_origin(band, wl_center)

            rows.append((setting, chip, odr, wl_center, sb, wl_src, tilt_src))
    tw.close()

# write markdown table
with open(BASE / 'tw_origin.md', 'w') as f:
    f.write('# _tw.fits provenance\n\n')
    f.write('Wavelength source: `vipere` = vipere telluric fit (prms<10, shift<10 km/s), '
            '`pipeline` = original cr2res wavelength calibration.\n\n')
    f.write('Tilt source: `vipere` = linear fit evaluated in region with good vipere tilt measurements, '
            '`interp` = interpolated across gap (no vipere data), '
            '`extrap` = extrapolated beyond measurement range.\n\n')
    f.write(f'Tilt gaps: ')
    for band in ['L', 'M']:
        gaps = tilt_gaps.get(band, [])
        if gaps:
            f.write(f'{band}: ' + ', '.join(f'{g[0]:.0f}-{g[1]:.0f} nm' for g in gaps) + '. ')
        else:
            f.write(f'{band}: none. ')
    f.write('\n\n')

    f.write('| setting | chip | order | wl_nm | tilt | wl_source | tilt_source |\n')
    f.write('|---------|------|-------|-------|------|-----------|-------------|\n')
    for setting, chip, odr, wl, tilt, wl_src, tilt_src in rows:
        f.write(f'| {setting} | {chip} | {odr} | {wl:.1f} | {tilt:+.5f} | {wl_src} | {tilt_src} |\n')

print(f'Wrote {len(rows)} rows to tw_origin.md')

# summary counts
from collections import Counter
wl_counts = Counter(r[5] for r in rows)
tilt_counts = Counter(r[6] for r in rows)
print(f'Wavelength: {dict(wl_counts)}')
print(f'Tilt: {dict(tilt_counts)}')
