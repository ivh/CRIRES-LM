#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "astropy"]
# ///
"""Compare measured A-B wavelength shifts from science data to slit tilt
predictions from _tw.fits SlitPolyB.

Reads tellfit_{A,B}.par.dat and tellfit_{A,B}_xcen.json sidecar files from
all reduction directories, computes median A-B wavelength difference per
(setting, chip, order) group, and compares to the shift predicted by the
tilt polynomial in the tracing tables.

Uses the actual nod throw from each observation's FITS header (ESO SEQ
NODTHROW) rather than a fixed value per setting.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits


MAX_ORDER = {
    'L3244': 7, 'L3262': 7, 'L3302': 7, 'L3340': 7,
    'L3377': 8, 'L3412': 8, 'L3426': 8,
    'M4187': 7, 'M4211': 8, 'M4266': 8, 'M4318': 8,
    'M4368': 8, 'M4416': 8, 'M4461': 8, 'M4504': 8, 'M4519': 9,
}

PIXEL_SCALE = 0.059  # arcsec/pixel

REFPIX = 1024
C_MS = 299792458.0


def parse_pardat(path):
    with open(path) as f:
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
    order_idx, det0 = divmod(vorder - 1, 3)
    chip = det0 + 1
    order_drs = max_order - order_idx
    return chip, order_drs


def eval_wl_at_pixel(wave0, wave1, wave2, xcen, pixel):
    dx = pixel - xcen
    return (wave0 + wave1 * dx + wave2 * dx * dx) / 10.0  # nm


def extract_setting(dirname):
    m = re.search(r'_([LM]\d{4})_', dirname)
    return m.group(1) if m else None


def get_nod_throw_pixels(dir_path):
    """Read nod throw from extractedA FITS header, convert to pixels."""
    ext_a = dir_path / 'cr2res_obs_nodding_extractedA.fits'
    if not ext_a.exists():
        return None
    try:
        h = fits.getheader(ext_a, 0)
        throw_arcsec = h.get('ESO SEQ NODTHROW')
        if throw_arcsec is None:
            return None
        return throw_arcsec / PIXEL_SCALE
    except Exception:
        return None


def collect_ab_diffs(reduced_dir, prms_max=10):
    """Collect per-observation A-B diffs with actual nod throw.

    Returns dict: (setting, chip, order) -> list of (diff_ms, throw_pix)
    """
    reduced = Path(reduced_dir)
    ab_diffs = defaultdict(list)
    n_files = 0
    n_no_throw = 0

    for parfile in sorted(reduced.glob('*/tellfit_A.par.dat')):
        dirname = parfile.parent.name
        setting = extract_setting(dirname)
        if not setting or setting not in MAX_ORDER:
            continue

        max_order = MAX_ORDER[setting]
        n_files += 1

        throw_pix = get_nod_throw_pixels(parfile.parent)
        if throw_pix is None:
            n_no_throw += 1
            continue

        obs_wl = {}
        for ab in ['A', 'B']:
            pf = parfile.parent / f'tellfit_{ab}.par.dat'
            xf = parfile.parent / f'tellfit_{ab}_xcen.json'
            if not pf.exists() or not xf.exists():
                continue
            with open(xf) as f:
                xmap = json.load(f)
            pars = parse_pardat(pf)

            for par in pars:
                vorder = int(par.get('order', 0))
                if vorder == 0:
                    continue
                prms = par.get('prms', -1)
                if prms < 0 or prms > prms_max:
                    continue
                chip, odrs = vipere_order_to_chip_order(vorder, max_order)
                xcen_key = f"{chip}_{odrs:02d}"
                if xcen_key not in xmap:
                    continue
                xcen = xmap[xcen_key]
                wave0 = par.get('wave0', np.nan)
                wave1 = par.get('wave1', np.nan)
                wave2 = par.get('wave2', np.nan)
                if any(np.isnan(v) for v in [wave0, wave1, wave2]):
                    continue
                wl = eval_wl_at_pixel(wave0, wave1, wave2, xcen, REFPIX)
                obs_wl[(chip, odrs, ab)] = wl

        nod_a = {(c, o): wl for (c, o, a), wl in obs_wl.items() if a == 'A'}
        nod_b = {(c, o): wl for (c, o, a), wl in obs_wl.items() if a == 'B'}
        for (chip, odrs) in nod_a:
            if (chip, odrs) in nod_b:
                wl_a = nod_a[(chip, odrs)]
                wl_b = nod_b[(chip, odrs)]
                wl_mean = (wl_a + wl_b) / 2
                diff_ms = (wl_a - wl_b) / wl_mean * C_MS
                ab_diffs[(setting, chip, odrs)].append((diff_ms, throw_pix))

    print(f"Scanned {n_files} directories ({n_no_throw} skipped, no throw header), "
          f"{len(ab_diffs)} groups with A-B pairs")
    return ab_diffs


def compute_ab_medians(ab_diffs):
    """Compute median A-B shift per group, using actual nod throw for prediction.

    Returns (medians, median_throws) dicts keyed by (setting, chip, order).
    """
    medians = {}
    median_throws = {}
    for key, vals_list in ab_diffs.items():
        diffs = np.array([v[0] for v in vals_list])
        throws = np.array([v[1] for v in vals_list])
        if len(diffs) < 5:
            continue
        med = np.median(diffs)
        mad = np.median(np.abs(diffs - med))
        good = np.abs(diffs - med) < 5 * max(mad, 1.0)
        if good.sum() < 5:
            continue
        medians[key] = np.median(diffs[good])
        median_throws[key] = np.median(throws[good])
    return medians, median_throws


def read_tw_predictions(ab_medians, median_throws):
    """Read SlitPolyB and Wavelength from _tw.fits, compute predicted A-B shift
    using the actual median nod throw per group."""
    predicted = {}
    tw_cache = {}

    for (setting, chip, odrs) in ab_medians:
        if setting not in tw_cache:
            tw_path = Path(f'{setting}_tw.fits')
            if not tw_path.exists():
                tw_cache[setting] = None
                continue
            tw_cache[setting] = fits.open(tw_path)

        hdul = tw_cache[setting]
        if hdul is None:
            continue
        ext = f'CHIP{chip}.INT1'
        if ext not in hdul:
            continue

        row = None
        for r in hdul[ext].data:
            if int(r['Order']) == odrs:
                row = r
                break
        if row is None:
            continue

        tilt = row['SlitPolyB'][0]
        wl_c = row['Wavelength']  # [c0, c1, c2] in nm
        wl_at_ref = wl_c[0] + wl_c[1] * REFPIX + wl_c[2] * REFPIX**2
        disp_at_ref = wl_c[1] + 2 * wl_c[2] * REFPIX  # nm/pixel

        dy = median_throws[(setting, chip, odrs)]
        pred_ms = tilt * dy * disp_at_ref / wl_at_ref * C_MS

        predicted[(setting, chip, odrs)] = {
            'pred_ms': pred_ms,
            'tilt': tilt,
            'wl_nm': wl_at_ref,
            'disp': disp_at_ref,
            'dy': dy,
        }

    for hdul in tw_cache.values():
        if hdul is not None:
            hdul.close()

    return predicted


def plot_comparison(ab_medians, tw_predictions):
    keys = sorted(set(ab_medians) & set(tw_predictions))
    if not keys:
        print("No matching keys for comparison")
        return

    measured = np.array([ab_medians[k] for k in keys])
    predicted = np.array([tw_predictions[k]['pred_ms'] for k in keys])
    wls = np.array([tw_predictions[k]['wl_nm'] for k in keys])

    resid = measured - predicted
    rms = np.std(resid)

    L = wls < 4200
    M = ~L

    fig, ax = plt.subplots(figsize=(7, 4))
    if L.any():
        ax.scatter(wls[L], resid[L], c='C0', s=30, alpha=0.7,
                   label='L band', edgecolors='none')
    if M.any():
        ax.scatter(wls[M], resid[M], c='C3', s=30, alpha=0.7,
                   label='M band', edgecolors='none')
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel('wavelength [nm]')
    ax.set_ylabel('measured - predicted A-B shift [m/s]')
    ax.set_title(f'Slit tilt validation ({len(keys)} groups, '
                 f'RMS={rms:.0f} m/s, median={np.median(resid):.0f} m/s)')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig('tilt_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n{'='*70}")
    print(f"  Tilt comparison: measured A-B vs _tw.fits prediction")
    print(f"{'='*70}")
    for k in sorted(keys, key=lambda k: tw_predictions[k]['wl_nm']):
        setting, chip, odrs = k
        idx = keys.index(k)
        print(f"  {setting} chip{chip} order{odrs:02d}: "
              f"wl={tw_predictions[k]['wl_nm']:7.1f} nm, "
              f"dy={tw_predictions[k]['dy']:5.1f} px, "
              f"predicted={predicted[idx]:+7.0f}, "
              f"measured={measured[idx]:+7.0f}, "
              f"resid={resid[idx]:+7.0f} m/s")
    print(f"\n  {len(keys)} groups, "
          f"residual RMS={rms:.0f} m/s, median={np.median(resid):.0f} m/s")
    print(f"\n  Wrote tilt_comparison.png")


def main():
    reduced_dir = Path('reduced')
    prms_max = 10
    if len(sys.argv) > 1:
        prms_max = float(sys.argv[1])

    print(f"Collecting A-B differences (prms < {prms_max})...")
    ab_diffs = collect_ab_diffs(reduced_dir, prms_max=prms_max)

    print("Computing median A-B shifts...")
    ab_medians, median_throws = compute_ab_medians(ab_diffs)
    print(f"  {len(ab_medians)} groups with >= 5 measurements")

    print("Reading _tw.fits predictions...")
    tw_pred = read_tw_predictions(ab_medians, median_throws)
    print(f"  {len(tw_pred)} groups matched to _tw.fits")

    plot_comparison(ab_medians, tw_pred)


if __name__ == '__main__':
    main()
