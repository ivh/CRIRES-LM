#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["astropy", "numpy", "matplotlib", "scipy"]
# ///
"""Measure slit tilt by cross-correlating A and B extracted spectra in sliding windows."""

import json
import numpy as np
from astropy.io import fits
from scipy.signal import correlate, medfilt
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
WINDOW = 400   # cross-correlation window size in pixels
STEP = 100     # step between windows
MAX_LAG = 15   # max shift to search (pixels)
MEDFILT_K = 81 # median filter kernel for continuum normalization


def normalize_spectrum(spec, kernel=MEDFILT_K):
    """Divide spectrum by running median to remove blaze/continuum shape."""
    s = spec.copy()
    bad = ~np.isfinite(s) | (s == 0)
    s[bad] = np.nanmedian(s[~bad]) if (~bad).any() else 1.0
    cont = medfilt(s, kernel_size=kernel)
    cont[cont == 0] = 1.0
    normed = spec / cont
    normed[bad] = np.nan
    return normed


def xcorr_shift(spec_a, spec_b, max_lag=MAX_LAG):
    """Cross-correlate two spectra, return sub-pixel shift of A relative to B.

    Positive shift = A is at higher pixel values than B.
    Sign convention matches vipere tilt: tilt = shift / dy.
    """
    n = len(spec_a)
    if n < 20:
        return np.nan, 0.0

    a = spec_a - np.nanmean(spec_a)
    b = spec_b - np.nanmean(spec_b)

    a = np.where(np.isfinite(a), a, 0)
    b = np.where(np.isfinite(b), b, 0)

    norm_a = np.sqrt(np.sum(a**2))
    norm_b = np.sqrt(np.sum(b**2))
    if norm_a == 0 or norm_b == 0:
        return np.nan, 0.0

    # correlate(a, b): peak at positive lag means a is shifted right relative to b
    cc = correlate(a, b, mode='full') / (norm_a * norm_b)
    lags = np.arange(-n + 1, n)

    mask = np.abs(lags) <= max_lag
    cc = cc[mask]
    lags = lags[mask]

    ipeak = np.argmax(cc)
    peak_cc = cc[ipeak]

    # parabolic interpolation for sub-pixel accuracy
    if 0 < ipeak < len(cc) - 1:
        y0, y1, y2 = cc[ipeak - 1], cc[ipeak], cc[ipeak + 1]
        denom = 2 * (2 * y1 - y0 - y2)
        if denom != 0:
            delta = (y0 - y2) / denom
        else:
            delta = 0
        shift = lags[ipeak] + delta
    else:
        shift = float(lags[ipeak])

    return float(shift), float(peak_cc)


def measure_tilt_xcorr(spec_a, spec_b, window=WINDOW, step=STEP):
    """Sliding-window cross-correlation on normalized spectra."""
    # normalize to remove blaze shape
    norm_a = normalize_spectrum(spec_a)
    norm_b = normalize_spectrum(spec_b)

    n = len(norm_a)
    centers = []
    shifts = []
    cc_peaks = []

    for start in range(0, n - window + 1, step):
        end = start + window
        win_a = norm_a[start:end]
        win_b = norm_b[start:end]

        good = np.isfinite(win_a) & np.isfinite(win_b)
        if good.sum() < window * 0.5:
            continue

        shift, cc_peak = xcorr_shift(win_a, win_b)
        if np.isfinite(shift) and cc_peak > 0.7:
            centers.append(start + window / 2)
            shifts.append(shift)
            cc_peaks.append(cc_peak)

    return np.array(centers), np.array(shifts), np.array(cc_peaks)


def process_setting(setting, dy, file_a=None, file_b=None):
    """Process one setting: cross-correlate A vs B for all orders on all chips."""
    if file_a is None:
        sdir = BASE / setting
        file_a = sdir / 'cr2res_obs_nodding_extractedA.fits'
        file_b = sdir / 'cr2res_obs_nodding_extractedB.fits'

    if not Path(file_a).exists() or not Path(file_b).exists():
        print(f'{setting}: missing files')
        return []

    hduA = fits.open(file_a)
    hduB = fits.open(file_b)

    results = []
    for chip in [1, 2, 3]:
        ext = f'CHIP{chip}.INT1'
        dataA = hduA[ext].data
        dataB = hduB[ext].data

        orders = sorted(set(int(c.split('_')[0])
                            for c in dataA.names if c.endswith('_SPEC')))

        for odr in orders:
            col_spec = f'{odr:02d}_01_SPEC'
            col_wl = f'{odr:02d}_01_WL'
            if col_spec not in dataA.names:
                continue

            specA = np.array(dataA[col_spec], dtype=np.float64)
            specB = np.array(dataB[col_spec], dtype=np.float64)

            centers, shifts, cc_peaks = measure_tilt_xcorr(specA, specB)
            if len(centers) == 0:
                continue

            tilt = -shifts / dy

            wlA = dataA[col_wl]
            wl_centers = np.interp(centers, np.arange(len(wlA)), wlA)

            for i in range(len(centers)):
                results.append({
                    'setting': setting,
                    'chip': chip,
                    'drs_order': odr,
                    'pixel': float(centers[i]),
                    'wl_nm': float(wl_centers[i]),
                    'shift_pix': float(shifts[i]),
                    'tilt': float(tilt[i]),
                    'cc_peak': float(cc_peaks[i]),
                })

    hduA.close()
    hduB.close()
    return results


def plot_comparison(results, vipere_meas, title, outfile):
    chip_colors = {1: 'C0', 2: 'C1', 3: 'C2'}
    chip_markers = {1: 'o', 2: 's', 3: 'D'}

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    ax = axes[0]
    for r in results:
        ax.plot(r['wl_nm'], r['tilt'],
                marker=chip_markers[r['chip']], color=chip_colors[r['chip']],
                ms=4, lw=0, alpha=0.6)
    if vipere_meas:
        for m in vipere_meas:
            ax.plot(m['wl_nm'], m['tilt'], marker='x', color='k', ms=6, lw=0, alpha=0.5)

    ax.set_ylabel('tilt (dx/dy)')
    ax.set_title(title)
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=chip_colors[c], marker=chip_markers[c], ms=5, lw=0)
               for c in [1, 2, 3]]
    labels = ['CHIP1 xcorr', 'CHIP2 xcorr', 'CHIP3 xcorr']
    if vipere_meas:
        handles.append(Line2D([0], [0], color='k', marker='x', ms=6, lw=0))
        labels.append('vipere')
    ax.legend(handles, labels, fontsize=8, loc='lower left')

    ax = axes[1]
    for chip in [1, 2, 3]:
        chip_orders = sorted(set(r['drs_order'] for r in results if r['chip'] == chip))
        for odr in chip_orders:
            subset = [r for r in results if r['chip'] == chip and r['drs_order'] == odr]
            pix = [r['pixel'] for r in subset]
            shift = [r['shift_pix'] for r in subset]
            label = f'CHIP{chip}' if odr == chip_orders[0] else None
            ax.plot(pix, shift, 'o-', color=chip_colors[chip], ms=3, lw=0.8, alpha=0.6,
                    label=label)
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Shift A vs B (pixels)')
    ax.set_title('Raw spectral shift vs pixel position')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2]
    for r in results:
        ax.plot(r['wl_nm'], r['cc_peak'],
                marker=chip_markers[r['chip']], color=chip_colors[r['chip']],
                ms=3, lw=0, alpha=0.5)
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('CC peak')
    ax.set_title('Cross-correlation peak quality')
    ax.axhline(0.9, color='gray', ls='--', lw=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f'Saved {outfile}')


def main():
    # load measured dy for telluric comparison
    with open(BASE / 'nod_throw_measurements.json') as f:
        nod_data = json.load(f)
    subset = [r for r in nod_data if r['setting'] == 'L3262' and r['dy_err'] < 0.2]
    dy_vals = [r['dy_pix'] for r in subset]
    med = np.median(dy_vals)
    mad = np.median([abs(v - med) for v in dy_vals])
    good = [v for v in dy_vals if abs(v - med) < max(5 * mad, 1.0)]
    dy_telluric = np.median(good)

    # load vipere tilt for comparison
    with open(BASE / 'tilt_fits.json') as f:
        vipere_data = json.load(f)
    vipere_meas = [m for m in vipere_data['measurements'] if m['setting'] == 'L3262']

    # --- 1. Telluric L3262 ---
    print(f'=== Telluric L3262: dy = {dy_telluric:.2f} pix ===')
    results_tel = process_setting('L3262', dy_telluric)
    print(f'{len(results_tel)} measurements')
    plot_comparison(results_tel, vipere_meas,
                    'L3262 telluric: xcorr (colored) vs vipere (black x)',
                    BASE / 'xcorr_tilt_L3262.png')

    # --- 2. Flats ---
    # slit fraction 0.1-0.3 and 0.7-0.9 -> separation = 0.6 * slit_length
    for setting in ['L3262', 'M4318']:
        flat_a = BASE.parent / 'flats' / f'{setting}_flatA.fits'
        flat_b = BASE.parent / 'flats' / f'{setting}_flatB.fits'
        if not flat_a.exists():
            continue

        tw = fits.open(BASE.parent / f'{setting}_tw.fits')
        tab = tw['CHIP2.INT1'].data
        row = tab[len(tab)//2]
        x = 1024
        y_upper = sum(c * x**j for j, c in enumerate(row['Upper']))
        y_lower = sum(c * x**j for j, c in enumerate(row['Lower']))
        slit_pix = y_upper - y_lower
        tw.close()
        dy_flat = 0.6 * slit_pix

        vipere_set = [m for m in vipere_data['measurements'] if m['setting'] == setting]
        print(f'\n=== Flat {setting}: slit = {slit_pix:.1f} pix, dy = {dy_flat:.1f} pix ===')

        results_flat = process_setting(setting, dy_flat,
                                       file_a=flat_a, file_b=flat_b)
        print(f'{len(results_flat)} measurements')
        plot_comparison(results_flat, vipere_set,
                        f'{setting} flat: xcorr (colored) vs vipere (black x), dy={dy_flat:.1f}',
                        BASE / f'xcorr_tilt_{setting}_flat.png')


if __name__ == '__main__':
    main()
