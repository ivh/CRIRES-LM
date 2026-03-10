#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["astropy", "numpy", "matplotlib", "scipy"]
# ///
"""Measure slit tilt using corrected nod throw, group by echelle order, fit smooth polynomials."""

import json
import numpy as np
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

BASE = Path(__file__).parent
N_SAMPLE = 10

# per-setting number of orders
N_ORDERS = {
    'L3244': 6, 'L3262': 6, 'L3302': 6, 'L3340': 6,
    'L3377': 7, 'L3412': 7, 'L3426': 7,
    'M4187': 6, 'M4211': 5, 'M4266': 7,
    'M4318': 6, 'M4368': 6, 'M4416': 6,
    'M4461': 5, 'M4504': 5,
}


def vipere_order_to_crires(order_idx, hdu):
    det = ((order_idx - 1) % 3) + 1
    spec_order = ((order_idx - 1) // 3)
    ext = f"CHIP{det}.INT1"
    data = hdu[ext].data
    n_orders = max(int(c.split('_')[0]) for c in data.names if c.endswith('_SPEC'))
    odr = n_orders - spec_order
    return det, odr


def reconstruct_wl(par_row, pixel_ok):
    xcen = np.nanmean(pixel_ok) + 18
    return np.poly1d([par_row["wave2"], par_row["wave1"], par_row["wave0"]])(pixel_ok - xcen)


def load_nod_throw_dy():
    """Load per-setting median dy from Gaussian fits."""
    with open(BASE / 'nod_throw_measurements.json') as f:
        data = json.load(f)

    dy_per_setting = {}
    for setting in N_ORDERS:
        subset = [r for r in data if r['setting'] == setting and r['dy_err'] < 0.2]
        if len(subset) < 3:
            continue
        dy_vals = [r['dy_pix'] for r in subset]
        med = np.median(dy_vals)
        mad = np.median([abs(v - med) for v in dy_vals])
        good = [v for v in dy_vals if abs(v - med) < max(5 * mad, 1.0)]
        dy_per_setting[setting] = np.median(good)

    return dy_per_setting


def collect_tilts(dy_per_setting):
    """Compute tilt for all orders in all settings using measured dy."""
    settings = sorted(d.name for d in BASE.iterdir() if d.is_dir()
                      and (d / "telluricA.par.dat").exists()
                      and (d / "telluricB.par.dat").exists()
                      and (d / "telluricB.par.dat").stat().st_size > 100)

    measurements = []
    for setting in settings:
        if setting not in dy_per_setting:
            print(f'{setting}: no dy measurement, skipping')
            continue
        delta_y = dy_per_setting[setting]
        sdir = BASE / setting

        parA = np.genfromtxt(sdir / "telluricA.par.dat", names=True, dtype=None, encoding=None)
        parB = np.genfromtxt(sdir / "telluricB.par.dat", names=True, dtype=None, encoding=None)
        if parA.ndim == 0: parA = parA.reshape(1)
        if parB.ndim == 0: parB = parB.reshape(1)

        hduA = fits.open(sdir / "cr2res_obs_nodding_extractedA.fits")
        hduB = fits.open(sdir / "cr2res_obs_nodding_extractedB.fits")

        dictA = {int(r["order"]): r for r in parA}
        dictB = {int(r["order"]): r for r in parB}

        for oi in sorted(set(dictA) & set(dictB)):
            rowA, rowB = dictA[oi], dictB[oi]
            prmsA, prmsB = rowA["prms"], rowB["prms"]
            if prmsA > 20 or prmsB > 20:
                continue

            det, odr = vipere_order_to_crires(oi, hduA)
            ext = f"CHIP{det}.INT1"
            dataA = hduA[ext].data
            col_wl = f"{odr:02d}_01_WL"
            col_spec = f"{odr:02d}_01_SPEC"
            if col_wl not in dataA.names:
                continue

            wlA_orig = dataA[col_wl]
            specA = dataA[col_spec]
            specB = hduB[ext].data[col_spec]
            wlB_orig = hduB[ext].data[col_wl]
            good = (np.isfinite(wlA_orig) & np.isfinite(specA) & (specA != 0) & (wlA_orig > 0)
                    & np.isfinite(wlB_orig) & np.isfinite(specB) & (specB != 0) & (wlB_orig > 0))
            if good.sum() < 50:
                continue

            pixel = np.arange(len(wlA_orig))
            pixel_ok = pixel[good]
            wlA = reconstruct_wl(rowA, pixel_ok)
            wlB = reconstruct_wl(rowB, pixel_ok)

            idx = np.linspace(0, len(pixel_ok) - 1, N_SAMPLE + 2, dtype=int)[1:-1]
            sample_pix = pixel_ok[idx]
            sample_wlA = wlA[idx]
            sample_wlB = wlB[idx]

            dwl_dx = np.gradient(wlA, pixel_ok)[idx]
            delta_wl = sample_wlA - sample_wlB
            delta_x = delta_wl / dwl_dx
            tilt = delta_x / delta_y

            if np.any(np.abs(tilt) > 0.5):
                continue

            wl_nm = sample_wlA / 10
            for i in range(len(idx)):
                measurements.append({
                    'setting': setting,
                    'chip': det,
                    'drs_order': odr,
                    'pixel': int(sample_pix[i]),
                    'wl_nm': float(wl_nm[i]),
                    'tilt': float(tilt[i]),
                    'prms_max': float(max(prmsA, prmsB)),
                })

        hduA.close()
        hduB.close()

    return measurements


def compute_echelle_m(settings_with_data):
    """Compute physical echelle order m for each (setting, DRS_order)."""
    # get center wavelengths per (setting, DRS_order) from extracted spectra
    seg_centers = {}
    for setting in settings_with_data:
        sdir = BASE / setting
        hdu = fits.open(sdir / 'cr2res_obs_nodding_extractedA.fits')
        for chip in [1, 2, 3]:
            ext = f'CHIP{chip}.INT1'
            data = hdu[ext].data
            for col in data.names:
                if col.endswith('_WL'):
                    odr = int(col.split('_')[0])
                    wl = data[col]
                    good = np.isfinite(wl) & (wl > 0)
                    if good.sum() > 10:
                        key = (setting, odr)
                        seg_centers.setdefault(key, []).append(np.mean(wl[good]))
        hdu.close()
    seg_centers = {k: np.mean(v) for k, v in seg_centers.items()}

    # for each setting, compute m from the two reddest DRS orders
    echelle_m = {}
    for setting in settings_with_data:
        orders = sorted(
            [(odr, seg_centers[(setting, odr)])
             for odr in set(k[1] for k in seg_centers if k[0] == setting)],
            key=lambda x: -x[1]  # reddest first
        )
        if len(orders) < 2:
            continue
        odr_red, wl_red = orders[0]
        odr_next, wl_next = orders[1]
        m_red = round(wl_next / (wl_red - wl_next))
        for odr, wl in orders:
            echelle_m[(setting, odr)] = m_red + (odr - odr_red)

    return echelle_m


def assign_echelle_orders(measurements):
    """Assign physical echelle order m and band to each measurement."""
    for m in measurements:
        m['band'] = 'L' if m['setting'].startswith('L') else 'M'

    settings_with_data = sorted(set(m['setting'] for m in measurements))
    echelle_m = compute_echelle_m(settings_with_data)

    for m in measurements:
        key = (m['setting'], m['drs_order'])
        m['echelle'] = echelle_m.get(key, -1)

    # summary
    by_echelle = defaultdict(list)
    for m in measurements:
        by_echelle[(m['band'], m['echelle'])].append(m)
    for (band, ech), meas in sorted(by_echelle.items()):
        wl_min = min(m['wl_nm'] for m in meas)
        wl_max = max(m['wl_nm'] for m in meas)
        settings = sorted(set(m['setting'] for m in meas))
        print(f'  m={ech} ({band}): {wl_min:.0f}-{wl_max:.0f} nm, '
              f'{len(meas)} pts, {len(settings)} settings: {", ".join(settings)}')

    return measurements


def fit_tilt_per_order(measurements):
    """Fit polynomial tilt(wavelength) per echelle order, separately for L and M."""
    by_group = defaultdict(list)
    for m in measurements:
        by_group[(m['band'], m['echelle'])].append(m)

    fits_result = {}
    for (band, echelle), meas in sorted(by_group.items()):
        wl = np.array([m['wl_nm'] for m in meas])
        tilt = np.array([m['tilt'] for m in meas])

        # sigma clip
        med = np.median(tilt)
        mad = np.median(np.abs(tilt - med))
        good = np.abs(tilt - med) < max(5 * mad, 0.01)
        wl_good = wl[good]
        tilt_good = tilt[good]

        if len(wl_good) < 5:
            print(f'  {band} m={echelle}: only {len(wl_good)} good points, skipping fit')
            continue

        # linear for narrow ranges, quadratic for wider
        wl_range = wl_good.max() - wl_good.min()
        deg = 1 if wl_range < 150 else 2
        coeff = np.polyfit(wl_good, tilt_good, deg)
        residuals = tilt_good - np.polyval(coeff, wl_good)
        rms = np.std(residuals)

        wl_min, wl_max = wl_good.min(), wl_good.max()
        n_settings = len(set(m['setting'] for m in meas))

        fits_result[(band, echelle)] = {
            'coeff': coeff, 'wl_min': wl_min, 'wl_max': wl_max,
            'n_pts': len(wl_good), 'n_settings': n_settings,
            'rms': rms, 'deg': deg,
        }
        print(f'  {band} m={echelle}: {wl_min:.0f}-{wl_max:.0f} nm, '
              f'{len(wl_good)} pts from {n_settings} settings, '
              f'deg={deg}, rms={rms:.5f}')

    return fits_result


def plot_results(measurements, fits_result):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    ax_data, ax_fit = axes

    chip_colors = {1: 'C0', 2: 'C1', 3: 'C2'}
    chip_markers = {1: 'o', 2: 's', 3: 'D'}

    for m in measurements:
        ax_data.plot(m['wl_nm'], m['tilt'],
                     marker=chip_markers[m['chip']], color=chip_colors[m['chip']],
                     ms=2, lw=0, alpha=0.3)

    ax_data.set_ylabel('tilt (dx/dy) [pix/pix]')
    ax_data.set_title('Slit tilt measurements (corrected nod throw)')
    ax_data.axhline(0, color='k', ls=':', lw=0.5)
    ax_data.grid(True, alpha=0.3)
    ax_data.set_ylim(-0.15, 0.05)

    # plot fits, colored by echelle order m
    cmap = plt.cm.viridis
    all_m = sorted(set(e for _, e in fits_result.keys()))
    m_norm = {m: i / max(len(all_m) - 1, 1) for i, m in enumerate(all_m)}

    for (band, echelle), fit in sorted(fits_result.items()):
        wl_plot = np.linspace(fit['wl_min'], fit['wl_max'], 100)
        tilt_plot = np.polyval(fit['coeff'], wl_plot)
        color = cmap(m_norm[echelle])
        ls = '-' if band == 'L' else '--'
        ax_fit.plot(wl_plot, tilt_plot, color=color, lw=2, alpha=0.8, ls=ls,
                    label=f'{band} m={echelle}')

    # overlay data
    for m in measurements:
        ax_fit.plot(m['wl_nm'], m['tilt'],
                    marker=chip_markers[m['chip']], color=chip_colors[m['chip']],
                    ms=1.5, lw=0, alpha=0.15)

    ax_fit.set_ylabel('tilt (dx/dy) [pix/pix]')
    ax_fit.set_xlabel('Wavelength [nm]')
    ax_fit.set_title('Polynomial fits per echelle order (solid=L, dashed=M)')
    ax_fit.axhline(0, color='k', ls=':', lw=0.5)
    ax_fit.grid(True, alpha=0.3)
    ax_fit.set_ylim(-0.15, 0.05)
    ax_fit.legend(fontsize=6, ncol=4, loc='lower left')

    from matplotlib.lines import Line2D
    chip_handles = [Line2D([0], [0], color=chip_colors[c], marker=chip_markers[c], ms=5, lw=0)
                    for c in [1, 2, 3]]
    ax_data.legend(chip_handles, ['CHIP1', 'CHIP2', 'CHIP3'], loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.savefig(BASE / 'tilt_fits.png', dpi=150)
    print(f'Saved tilt_fits.png')


def main():
    print('Loading measured nod throws...')
    dy = load_nod_throw_dy()
    for s in sorted(dy):
        print(f'  {s}: dy = {dy[s]:.2f} pix')

    print('\nCollecting tilt measurements...')
    measurements = collect_tilts(dy)
    print(f'  {len(measurements)} raw measurements')

    print('\nAssigning echelle orders...')
    measurements = assign_echelle_orders(measurements)
    echelle_ids = sorted(set((m['band'], m['echelle']) for m in measurements))
    print(f'  {len(echelle_ids)} groups (band, m)')

    print('\nFitting tilt per echelle order...')
    fits_result = fit_tilt_per_order(measurements)

    plot_results(measurements, fits_result)

    # save everything
    out = {
        'dy_per_setting': dy,
        'measurements': measurements,
        'fits': {f'{b}_m{e}': {'coeff': list(f['coeff']), 'wl_min': f['wl_min'],
                                'wl_max': f['wl_max'], 'rms': f['rms']}
                 for (b, e), f in fits_result.items()},
    }
    with open(BASE / 'tilt_fits.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f'Saved tilt_fits.json')


if __name__ == '__main__':
    main()
