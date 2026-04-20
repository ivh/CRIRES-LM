#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "astropy"]
# ///
"""Compare _tw.fits wavelength polynomials against median vipere science solutions.

One plot per setting showing velocity offset as a function of wavelength,
with each (chip, order) trace as a short curve segment.
"""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

_u = lambda n: {1: n, 2: n, 3: n}
MAX_ORDER = {
    'L3244': _u(7), 'L3262': _u(7), 'L3302': _u(7), 'L3340': _u(7),
    'L3377': _u(8), 'L3412': _u(8), 'L3426': _u(8),
    'M4187': _u(7), 'M4211': _u(7), 'M4266': _u(8),
    'M4318': _u(8), 'M4368': _u(8), 'M4416': _u(8),
    'M4461': _u(8), 'M4504': _u(8), 'M4519': _u(9),
}

C_MS = 299792458.0
C_KMS = C_MS / 1000

FIT_EXCLUDE = {
    ('M4416', 1, 8),
    ('M4416', 1, 7),
}


def apply_velocity_correction(c0, c1, c2, a, b):
    """Correct wavelength poly for linear velocity offset dv = a + b*wl (km/s).

    Returns corrected (c0, c1, c2) such that wl_new(x) = wl(x) * (1 - dv(wl)/c).
    """
    alpha = a / C_KMS
    beta = b / C_KMS
    c0_new = c0 * (1 - alpha) - beta * c0**2
    c1_new = c1 * (1 - alpha) - beta * 2 * c0 * c1
    c2_new = c2 * (1 - alpha) - beta * (c1**2 + 2 * c0 * c2)
    return c0_new, c1_new, c2_new


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


def vipere_order_to_chip_order(vorder, max_per_chip):
    order_idx, det0 = divmod(vorder - 1, 3)
    chip = det0 + 1
    return chip, max_per_chip[chip] - order_idx


def extract_setting(dirname):
    m = re.search(r'_([LM]\d{4})_', dirname)
    return m.group(1) if m else None


def parse_tw_origin(path='tellurics/tw_origin.md'):
    """Return set of (setting, chip, order) that have pipeline wavelengths."""
    pipeline = set()
    p = Path(path)
    if not p.exists():
        return pipeline
    for line in p.read_text().splitlines():
        if '| pipeline |' not in line:
            continue
        parts = [x.strip() for x in line.split('|')]
        if len(parts) < 8:
            continue
        try:
            setting = parts[1]
            chip = int(parts[2])
            order = int(parts[3])
            if parts[6] == 'pipeline':
                pipeline.add((setting, chip, order))
        except (ValueError, IndexError):
            continue
    return pipeline


def collect_science_polys(reduced_dir, prms_max=10):
    """Collect vipere polynomials converted to _tw format (nm, pixel-centered)."""
    reduced = Path(reduced_dir)
    poly_data = defaultdict(list)
    n_dirs = 0

    for parfile in sorted(reduced.glob('*/tellfit_A.par.dat')):
        dirname = parfile.parent.name
        setting = extract_setting(dirname)
        if not setting or setting not in MAX_ORDER:
            continue
        max_order = MAX_ORDER[setting]
        n_dirs += 1

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
                w0 = par.get('wave0', np.nan)
                w1 = par.get('wave1', np.nan)
                w2 = par.get('wave2', np.nan)
                if any(np.isnan(v) for v in [w0, w1, w2]):
                    continue

                c0 = (w0 - w1 * xcen + w2 * xcen**2) / 10
                c1 = (w1 - 2 * w2 * xcen) / 10
                c2 = w2 / 10
                poly_data[(setting, chip, odrs)].append([c0, c1, c2])

    print(f"Scanned {n_dirs} directories")
    return poly_data


def read_tw(setting):
    tw_path = Path(f'tw_comp/{setting}_tw.fits')
    if not tw_path.exists():
        return {}
    result = {}
    with fits.open(tw_path) as hdul:
        for chip in [1, 2, 3]:
            ext = f'CHIP{chip}.INT1'
            if ext not in hdul:
                continue
            for row in hdul[ext].data:
                result[(chip, row['Order'])] = np.array(row['Wavelength'],
                                                        dtype=float)
    return result


def main():
    pipeline_orders = parse_tw_origin()
    print(f"Pipeline-wavelength orders in tw_origin: {len(pipeline_orders)}")

    print("Collecting science vipere wavelengths (prms < 10)...")
    poly_data = collect_science_polys('reduced')

    settings = sorted(set(s for s, c, o in poly_data))
    print(f"Settings with data: {', '.join(settings)}\n")

    pixels = np.linspace(0, 2047, 200)
    chip_colors = {1: 'C0', 2: 'C1', 3: 'C2'}
    all_fits = {}

    plt.rcParams.update({
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })

    for setting in settings:
        tw = read_tw(setting)
        if not tw:
            print(f"{setting}: no _tw.fits\n")
            continue

        traces = []
        for (s, chip, order), obs_list in poly_data.items():
            if s != setting or (chip, order) not in tw:
                continue
            obs_arr = np.array(obs_list)
            n_obs = len(obs_arr)
            if n_obs < 3:
                continue

            med_c = np.median(obs_arr, axis=0)
            tw_c = tw[(chip, order)]

            wl_1024_all = obs_arr[:, 0] + obs_arr[:, 1] * 1024 + obs_arr[:, 2] * 1024**2
            med_1024 = np.median(wl_1024_all)
            mad_1024 = np.median(np.abs(wl_1024_all - med_1024))
            unc_nm = 1.4826 * mad_1024 / np.sqrt(n_obs)
            unc_kms = unc_nm / med_1024 * C_MS / 1000

            tw_1024 = tw_c[0] + tw_c[1] * 1024 + tw_c[2] * 1024**2
            offset_1024 = (tw_1024 - med_1024) / med_1024 * C_MS / 1000

            wl_sci = med_c[0] + med_c[1] * pixels + med_c[2] * pixels**2
            wl_tw_px = tw_c[0] + tw_c[1] * pixels + tw_c[2] * pixels**2
            offset_curve = (wl_tw_px - wl_sci) / wl_sci * C_MS / 1000

            is_pipeline = (setting, chip, order) in pipeline_orders

            traces.append({
                'chip': chip, 'order': order, 'n': n_obs,
                'wl_1024': med_1024,
                'offset_1024': offset_1024,
                'unc_kms': unc_kms,
                'mad_kms': 1.4826 * mad_1024 / med_1024 * C_MS / 1000,
                'wl_curve': wl_sci,
                'offset_curve': offset_curve,
                'pipeline': is_pipeline,
            })

        if not traces:
            print(f"{setting}: no usable traces\n")
            continue

        traces.sort(key=lambda t: t['wl_1024'])

        # fit line through vipere-sourced traces (iterative 3-sigma clip)
        fit_traces = [t for t in traces if not t['pipeline']
                      and (setting, t['chip'], t['order']) not in FIT_EXCLUDE]
        if len(fit_traces) >= 2:
            wl_fit = np.array([t['wl_1024'] for t in fit_traces])
            dv_fit = np.array([t['offset_1024'] for t in fit_traces])
            mask = np.ones(len(wl_fit), dtype=bool)
            for _ in range(3):
                if mask.sum() < 2:
                    break
                p = np.polyfit(wl_fit[mask], dv_fit[mask], 1)
                resid = dv_fit - np.polyval(p, wl_fit)
                mad = np.median(np.abs(resid[mask]))
                mask = np.abs(resid) < 3 * 1.4826 * mad
            fit_slope, fit_intercept = p
            fit_rms = np.sqrt(np.mean(resid[mask]**2))
            n_used = mask.sum()
            n_rejected = len(mask) - n_used
        else:
            fit_slope = fit_intercept = fit_rms = np.nan
            n_used = len(fit_traces)
            n_rejected = 0

        all_fits[setting] = {
            'intercept': fit_intercept, 'slope': fit_slope,
            'rms': fit_rms, 'n_used': n_used, 'n_rejected': n_rejected,
            'traces': traces,
            'wl_range': (min(t['wl_1024'] for t in traces),
                         max(t['wl_1024'] for t in traces)),
        }

        # print summary
        print(f"{'='*70}")
        print(f"  {setting}")
        print(f"{'='*70}")
        if not np.isnan(fit_slope):
            print(f"  FIT: offset = {fit_intercept:+.3f} + {fit_slope:+.6f} * wl_nm  "
                  f"(rms={fit_rms:.2f} km/s, {n_used} pts, {n_rejected} rejected)")
        else:
            print(f"  FIT: insufficient data ({n_used} vipere traces)")

        for t in traces:
            src = "PIPE" if t['pipeline'] else "vipe"
            resid_from_fit = t['offset_1024'] - (fit_intercept + fit_slope * t['wl_1024']) if not np.isnan(fit_slope) else np.nan
            flag = ""
            if t['pipeline'] and abs(t['offset_1024']) > 3:
                flag = " ***"
            elif not t['pipeline'] and abs(resid_from_fit) > 1:
                flag = " *"
            print(f"  CHIP{t['chip']} o{t['order']:02d} [{src}]: "
                  f"wl={t['wl_1024']:7.1f} nm, "
                  f"offset={t['offset_1024']:+8.2f} km/s, "
                  f"resid={resid_from_fit:+6.2f} km/s, "
                  f"n={t['n']:5d}, "
                  f"MAD={t['mad_kms']:5.2f} km/s"
                  f"{flag}")

        # note tw orders with no or too few science measurements
        for (chip, order) in sorted(tw):
            if not any(t['chip'] == chip and t['order'] == order for t in traces):
                wl = tw[(chip, order)][0] + tw[(chip, order)][1] * 1024 + tw[(chip, order)][2] * 1024**2
                n_raw = len(poly_data.get((setting, chip, order), []))
                src = "PIPE" if (setting, chip, order) in pipeline_orders else "vipe"
                print(f"  CHIP{chip} o{order:02d} [{src}]: "
                      f"wl={wl:7.1f} nm, "
                      f"NO DATA (n={n_raw})")

        # plot
        fig, ax = plt.subplots(figsize=(7, 2.7))
        chip_plotted = {}
        labeled_orders = set()

        for t in traces:
            chip = t['chip']
            color = chip_colors[chip]
            ls = ':' if t['pipeline'] else '-'
            lw = 2.0 if t['pipeline'] else 1.2

            if chip not in chip_plotted:
                chip_plotted[chip] = ax.plot(
                    t['wl_curve'], t['offset_curve'],
                    color=color, ls=ls, lw=lw, label=f'CHIP{chip}')[0]
            else:
                ax.plot(t['wl_curve'], t['offset_curve'],
                        color=color, ls=ls, lw=lw)

            ax.fill_between(t['wl_curve'],
                            t['offset_curve'] - t['unc_kms'],
                            t['offset_curve'] + t['unc_kms'],
                            color=color, alpha=0.10)

            # label once per order, prefer chip 2
            if t['order'] not in labeled_orders and t['chip'] == 2:
                mid = len(t['wl_curve']) // 2
                ax.annotate(
                    f"o{t['order']}", xy=(t['wl_curve'][mid], t['offset_curve'][mid]),
                    fontsize=10, ha='center', va='bottom', color='k', alpha=0.7,
                    xytext=(0, 3), textcoords='offset points')
                labeled_orders.add(t['order'])

        # second pass: label orders that had no chip-2 data
        for t in traces:
            if t['order'] not in labeled_orders:
                mid = len(t['wl_curve']) // 2
                ax.annotate(
                    f"o{t['order']}", xy=(t['wl_curve'][mid], t['offset_curve'][mid]),
                    fontsize=10, ha='center', va='bottom', color='k', alpha=0.7,
                    xytext=(0, 3), textcoords='offset points')
                labeled_orders.add(t['order'])

        # fit line
        if not np.isnan(fit_slope):
            wl_range = np.array([min(t['wl_curve'][0] for t in traces),
                                 max(t['wl_curve'][-1] for t in traces)])
            fit_line_y = fit_intercept + fit_slope * wl_range
            ax.plot(wl_range, fit_line_y, 'r-', lw=1.5, alpha=0.8, zorder=5)

        ax.axhline(0, color='k', lw=0.5)
        ax.axhline(3, color='gray', lw=0.5, ls=':', alpha=0.5)
        ax.axhline(-3, color='gray', lw=0.5, ls=':', alpha=0.5)

        ax.set_xlabel('wavelength (nm)')
        ax.set_ylabel(r'$\Delta v$ (km/s)')
        ax.set_title(f'{setting}: _tw.fits vs aggregate science wavelength')

        fig.tight_layout()
        outfile = f'tw_comparison_{setting}.png'
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> {outfile}\n")

    # --- apply corrections to _tw.fits ---
    print("\n" + "=" * 70)
    print("  Applying velocity corrections to _tw.fits")
    print("=" * 70)
    for setting in sorted(all_fits):
        f = all_fits[setting]
        if np.isnan(f['slope']):
            print(f"  {setting}: SKIPPED (no fit)")
            continue
        a, b = f['intercept'], f['slope']
        src_path = Path(f'tw_comp/{setting}_tw.fits')
        dst_path = Path(f'{setting}_tw.fits')
        if not src_path.exists():
            continue
        # always read from the pre-correction baseline in tw_comp/ and write
        # the corrected result to cwd — script is idempotent
        shutil.copyfile(src_path, dst_path)
        with fits.open(dst_path, mode='update') as hdul:
            for chip in [1, 2, 3]:
                ext = f'CHIP{chip}.INT1'
                if ext not in hdul:
                    continue
                for row in hdul[ext].data:
                    c = np.array(row['Wavelength'], dtype=float)
                    wl_old = c[0] + c[1] * 1024 + c[2] * 1024**2
                    c0n, c1n, c2n = apply_velocity_correction(c[0], c[1], c[2], a, b)
                    wl_new = c0n + c1n * 1024 + c2n * 1024**2
                    dv = (wl_old - wl_new) / wl_new * C_KMS
                    print(f"  {setting} CHIP{chip} o{row['Order']:02d}: "
                          f"dv={dv:+.2f} km/s  "
                          f"wl@1024: {wl_old:.4f} -> {wl_new:.4f} nm")
                    row['Wavelength'][:] = [c0n, c1n, c2n]
        print(f"  -> {dst_path} updated\n")

    # --- paper figure: all fits in one panel ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    band_colors = {'L': '#2166ac', 'M': '#b2182b'}
    # slight shade variation within each band
    l_settings = [s for s in sorted(all_fits) if s.startswith('L')]
    m_settings = [s for s in sorted(all_fits) if s.startswith('M')]

    def shade(base_hex, idx, n):
        import matplotlib.colors as mcolors
        r, g, b_ = mcolors.to_rgb(base_hex)
        f = 0.5 + 0.5 * idx / max(n - 1, 1)
        return (r * f, g * f, b_ * f)

    label_y_offsets = {}

    for band, slist, base in [('L', l_settings, band_colors['L']),
                               ('M', m_settings, band_colors['M'])]:
        for i, setting in enumerate(slist):
            f = all_fits.get(setting)
            if f is None or np.isnan(f['slope']):
                continue
            color = shade(base, i, len(slist))
            a, b = f['intercept'], f['slope']

            for t in f['traces']:
                if not t['pipeline'] and (setting, t['chip'], t['order']) not in FIT_EXCLUDE:
                    ax.plot(t['wl_1024'], t['offset_1024'], '.', color=color,
                            ms=2.5, alpha=0.5, zorder=3)

            wl_ends = np.array(f['wl_range'])
            dv_ends = a + b * wl_ends
            ax.plot(wl_ends, dv_ends, '-', color=color, lw=1.0, alpha=0.85, zorder=4)

            # label at left end for L, right end for M to reduce overlap
            if band == 'L':
                lx, ly = wl_ends[0], dv_ends[0]
                ha, xoff = 'right', -3
            else:
                lx, ly = wl_ends[1], dv_ends[1]
                ha, xoff = 'left', 3

            # nudge labels vertically to avoid overlap
            for prev_y in label_y_offsets.get((band, ha), []):
                if abs(ly - prev_y) < 0.6:
                    ly += 0.6 if ly >= prev_y else -0.6
            label_y_offsets.setdefault((band, ha), []).append(ly)

            ax.annotate(setting, xy=(lx, ly),
                        fontsize=5.5, color=color, fontweight='bold',
                        xytext=(xoff, 0), textcoords='offset points',
                        va='center', ha=ha)

    ax.axhline(0, color='k', lw=0.5, zorder=1)
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('offset: _tw $-$ science median (km/s)')
    fig.tight_layout()
    paper_file = 'tw_correction_summary.png'
    fig.savefig(paper_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  -> {paper_file}")


if __name__ == '__main__':
    main()
