#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Measure vipere wavelength precision from inter-chip and inter-order gap consistency.

For a given echelle order, the 3 CRIRES+ chips are rigidly mounted, so
the wavelength gap between chip centers is a physical constant. Similarly,
adjacent orders on the same chip have a fixed wavelength separation.

By comparing these gaps across thousands of science observations, the median
gives the "true" gap and the scatter gives the wavelength precision.

Three complementary metrics:
1. Absolute scatter: RMS of wavelength at pixel 1024 per (setting, chip, order)
2. Inter-chip gap scatter: consistency between chips for same echelle order
3. Inter-order gap scatter: consistency between adjacent orders on same chip
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


ORDERS_PER_CHIP = {
    'L3244': 6, 'L3262': 6, 'L3302': 6, 'L3340': 6,
    'L3377': 7, 'L3412': 7, 'L3426': 7,
    'M4187': 6, 'M4211': 5, 'M4266': 7, 'M4318': 6,
    'M4368': 6, 'M4416': 6, 'M4461': 5, 'M4504': 5, 'M4519': 7,
}

# max DRS order number per setting (from _tw.fits Order column)
MAX_ORDER = {
    'L3244': 7, 'L3262': 7, 'L3302': 7, 'L3340': 7,
    'L3377': 8, 'L3412': 8, 'L3426': 8,
    'M4187': 7, 'M4211': 8, 'M4266': 8, 'M4318': 8,
    'M4368': 8, 'M4416': 8, 'M4461': 8, 'M4504': 8, 'M4519': 9,
}

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


def collect_paired(reduced_dir, prms_max=10):
    """Collect per-observation wavelengths at REFPIX, keeping A+B from same
    observation together."""
    reduced = Path(reduced_dir)
    results = []
    n_files = 0

    for parfile in sorted(reduced.glob('*/tellfit_A.par.dat')):
        dirname = parfile.parent.name
        setting = extract_setting(dirname)
        if not setting or setting not in MAX_ORDER:
            continue

        max_order = MAX_ORDER[setting]
        obs_wl = {}
        n_files += 1

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

        if obs_wl:
            results.append((dirname, setting, obs_wl))

    print(f"Scanned {n_files} directories, "
          f"{len(results)} with usable measurements (prms < {prms_max})")
    return results


def compute_all_metrics(paired_data):
    """Compute absolute scatter, inter-chip gaps, inter-order gaps, A-B diffs."""
    # absolute wavelength per (setting, chip, order)
    abs_wl = defaultdict(list)
    # inter-chip gaps: (setting, order, c_lo, c_hi)
    chip_gaps = defaultdict(list)
    # inter-order gaps: (setting, chip, o_lo, o_hi)
    order_gaps = defaultdict(list)
    # A-B wavelength difference per segment: (setting, chip, order)
    ab_diffs = defaultdict(list)

    for dirname, setting, obs_wl in paired_data:
        for ab in ['A', 'B']:
            nod = {(c, o): wl for (c, o, a), wl in obs_wl.items() if a == ab}
            if not nod:
                continue

            for (chip, odrs), wl in nod.items():
                abs_wl[(setting, chip, odrs)].append(wl)

            orders = set(o for c, o in nod)
            for odrs in orders:
                for c_lo, c_hi in [(1, 2), (2, 3)]:
                    if (c_lo, odrs) in nod and (c_hi, odrs) in nod:
                        wl_lo = nod[(c_lo, odrs)]
                        wl_hi = nod[(c_hi, odrs)]
                        gap_nm = wl_hi - wl_lo
                        wl_mean = (wl_lo + wl_hi) / 2
                        gap_ms = gap_nm / wl_mean * C_MS
                        chip_gaps[(setting, odrs, c_lo, c_hi)].append(gap_ms)

            chips = set(c for c, o in nod)
            for chip in chips:
                chip_orders = sorted(o for c, o in nod if c == chip)
                for i in range(len(chip_orders) - 1):
                    o_lo, o_hi = chip_orders[i], chip_orders[i + 1]
                    if o_hi - o_lo != 1:
                        continue
                    wl_lo = nod[(chip, o_lo)]
                    wl_hi = nod[(chip, o_hi)]
                    gap_nm = wl_hi - wl_lo
                    wl_mean = (wl_lo + wl_hi) / 2
                    gap_ms = gap_nm / wl_mean * C_MS
                    order_gaps[(setting, chip, o_lo, o_hi)].append(gap_ms)

        # A-B difference: same segment, both nods fitted in this observation
        nod_a = {(c, o): wl for (c, o, a), wl in obs_wl.items() if a == 'A'}
        nod_b = {(c, o): wl for (c, o, a), wl in obs_wl.items() if a == 'B'}
        for (chip, odrs) in nod_a:
            if (chip, odrs) in nod_b:
                wl_a = nod_a[(chip, odrs)]
                wl_b = nod_b[(chip, odrs)]
                diff_nm = wl_a - wl_b
                wl_mean = (wl_a + wl_b) / 2
                diff_ms = diff_nm / wl_mean * C_MS
                ab_diffs[(setting, chip, odrs)].append(diff_ms)

    return abs_wl, chip_gaps, order_gaps, ab_diffs


def group_scatter(vals_arr):
    """Compute scatter with 5-MAD sigma clipping. Returns (rms, mad, n_clean)."""
    vals = np.array(vals_arr)
    if len(vals) < 5:
        return np.nan, np.nan, 0
    med = np.median(vals)
    devs = vals - med
    mad = np.median(np.abs(devs))
    good = np.abs(devs) < 5 * max(mad, 1.0)
    devs_clean = devs[good]
    if len(devs_clean) < 5:
        return np.nan, np.nan, 0
    return np.std(devs_clean), mad, len(devs_clean)


def analyze_absolute(abs_wl):
    """Analyze absolute wavelength scatter per group. Returns (wl_centers, rms_ms, counts)."""
    print(f"\n{'='*70}")
    print(f"  Absolute wavelength scatter at pixel {REFPIX}")
    print(f"{'='*70}")

    wl_centers = []
    rms_vals = []
    mad_vals = []
    counts = []
    labels = []

    for key in sorted(abs_wl, key=lambda k: np.median(abs_wl[k])):
        setting, chip, odrs = key
        vals = np.array(abs_wl[key])
        rms, mad, n = group_scatter(vals)
        if np.isnan(rms):
            continue

        wl_med = np.median(vals)
        rms_ms = rms / wl_med * C_MS
        mad_ms = mad / wl_med * C_MS

        wl_centers.append(wl_med)
        rms_vals.append(rms_ms)
        mad_vals.append(mad_ms)
        counts.append(n)
        labels.append(f"{setting} c{chip} o{odrs:02d}")

        print(f"  {setting} chip{chip} order{odrs:02d}: "
              f"wl={wl_med:7.1f} nm, n={n:4d}, "
              f"RMS={rms_ms:7.0f} m/s, 1.48*MAD={1.4826*mad_ms:7.0f} m/s")

    wl_centers = np.array(wl_centers)
    rms_vals = np.array(rms_vals)
    mad_vals = np.array(mad_vals)
    counts = np.array(counts)

    # summary: separate L and M, and compute weighted stats
    for band, wl_lo, wl_hi in [('L', 2800, 4200), ('M', 4200, 5600), ('all', 2800, 5600)]:
        sel = (wl_centers >= wl_lo) & (wl_centers < wl_hi)
        if sel.sum() == 0:
            continue
        # only include groups with "reasonable" scatter (< 5 km/s)
        ok = sel & (rms_vals < 5000)
        if ok.sum() == 0:
            continue
        wt = counts[ok]
        med_rms = np.median(rms_vals[ok])
        wtd_rms = np.sqrt(np.average(rms_vals[ok]**2, weights=wt))
        med_mad = np.median(1.4826 * mad_vals[ok])
        print(f"\n  {band}-band ({ok.sum()}/{sel.sum()} groups with RMS<5km/s): "
              f"median RMS={med_rms:.0f} m/s, "
              f"weighted RMS={wtd_rms:.0f} m/s, "
              f"median 1.48*MAD={med_mad:.0f} m/s")

    return wl_centers, rms_vals, mad_vals, counts, labels


def analyze_gaps(gaps, label=""):
    """Compute scatter for gap groups. Returns (wl_centers, rms_ms, counts)."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    wl_centers = []
    rms_vals = []
    counts = []

    for key in sorted(gaps):
        vals = np.array(gaps[key])
        rms, mad, n = group_scatter(vals)
        if np.isnan(rms):
            continue
        rms_vals.append(rms)
        counts.append(n)

        # estimate center wavelength from gap median
        # (not exact, but directionally correct for plotting)
        # just use the key info
        if len(key) == 4 and isinstance(key[2], int) and key[2] <= 3:
            # inter-chip: (setting, order, c_lo, c_hi)
            setting, odrs, c_lo, c_hi = key
            label_str = f"{setting} o{odrs:02d} c{c_lo}-{c_hi}"
        else:
            # inter-order: (setting, chip, o_lo, o_hi)
            setting, chip, o_lo, o_hi = key
            label_str = f"{setting} c{chip} o{o_lo}-{o_hi}"

        print(f"  {label_str}: n={n:4d}, "
              f"RMS={rms:7.0f} m/s, 1.48*MAD={1.4826*mad:7.0f} m/s")

    rms_vals = np.array(rms_vals)
    counts = np.array(counts)

    if len(rms_vals) > 0:
        ok = rms_vals < 5000
        if ok.sum() > 0:
            med = np.median(rms_vals[ok])
            wtd = np.sqrt(np.average(rms_vals[ok]**2, weights=counts[ok]))
            print(f"\n  {ok.sum()}/{len(rms_vals)} groups with RMS<5km/s: "
                  f"median={med:.0f} m/s, weighted={wtd:.0f} m/s")

    return rms_vals, counts


def analyze_ab_diffs(ab_diffs):
    """Analyze A-B wavelength differences per segment."""
    print(f"\n{'='*70}")
    print(f"  A-B wavelength difference (within segment)")
    print(f"{'='*70}")

    wl_centers = []
    rms_vals = []
    mad_vals = []
    counts = []
    median_offsets = []

    for key in sorted(ab_diffs, key=lambda k: np.median(
            [abs(v) for v in ab_diffs[k]])):
        setting, chip, odrs = key
        vals = np.array(ab_diffs[key])
        rms, mad, n = group_scatter(vals)
        if np.isnan(rms):
            continue

        med = np.median(vals)
        # estimate wl from abs_wl if available, else use a dummy
        # (we know A-B keys match abs_wl keys)
        wl_centers.append(0)  # filled below
        rms_vals.append(rms)
        mad_vals.append(mad)
        counts.append(n)
        median_offsets.append(med)

        print(f"  {setting} chip{chip} order{odrs:02d}: "
              f"n={n:4d}, median A-B={med:+7.0f} m/s, "
              f"RMS={rms:7.0f} m/s, 1.48*MAD={1.4826*mad:7.0f} m/s")

    rms_vals = np.array(rms_vals)
    counts = np.array(counts)

    if len(rms_vals) > 0:
        ok = rms_vals < 5000
        if ok.sum() > 0:
            med = np.median(rms_vals[ok])
            wtd = np.sqrt(np.average(rms_vals[ok]**2, weights=counts[ok]))
            print(f"\n  {ok.sum()}/{len(rms_vals)} groups with RMS<5km/s: "
                  f"median={med:.0f} m/s, weighted={wtd:.0f} m/s")

    return rms_vals, counts


def plot_results(abs_data, chip_rms, order_rms, ab_rms):
    wl_centers, abs_rms_arr, abs_mad, abs_counts, abs_labels = abs_data

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # 1. Absolute scatter vs wavelength
    ax = axes[0, 0]
    L = wl_centers < 4200
    M = ~L
    if L.any():
        ax.scatter(wl_centers[L], abs_rms_arr[L], s=abs_counts[L] * 0.3 + 5,
                   c='C0', alpha=0.7, label='L band', edgecolors='none')
    if M.any():
        ax.scatter(wl_centers[M], abs_rms_arr[M], s=abs_counts[M] * 0.3 + 5,
                   c='C3', alpha=0.7, label='M band', edgecolors='none')
    ax.set_yscale('log')
    ax.set_ylim(100, 200000)
    ax.axhline(1000, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(500, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('wavelength [nm]')
    ax.set_ylabel('RMS scatter [m/s]')
    ax.set_title('Absolute wavelength repeatability')
    ax.legend(fontsize=8)

    # 2. Histogram of absolute scatter (good groups only)
    ax = axes[0, 1]
    ok = abs_rms_arr < 5000
    if ok.any():
        bins = np.linspace(0, 5000, 50)
        ax.hist(abs_rms_arr[ok & L], bins=bins, color='C0', alpha=0.6,
                label=f'L band (n={np.sum(ok & L)})', edgecolor='none')
        ax.hist(abs_rms_arr[ok & M], bins=bins, color='C3', alpha=0.6,
                label=f'M band (n={np.sum(ok & M)})', edgecolor='none')
        ax.axvline(np.median(abs_rms_arr[ok]), color='k', ls='--', lw=1,
                   label=f'median={np.median(abs_rms_arr[ok]):.0f} m/s')
        ax.set_xlabel('RMS scatter [m/s]')
        ax.set_ylabel('count (groups)')
        ax.set_title('Absolute scatter distribution (< 5 km/s)')
        ax.legend(fontsize=8)

    # 3. A-B scatter distribution
    ax = axes[0, 2]
    ok = ab_rms < 5000
    if ok.any():
        bins = np.linspace(0, 5000, 40)
        ax.hist(ab_rms[ok], bins=bins, color='C2', alpha=0.7, edgecolor='none')
        ax.axvline(np.median(ab_rms[ok]), color='k', ls='--', lw=1,
                   label=f'median={np.median(ab_rms[ok]):.0f} m/s')
        ax.set_xlabel('RMS scatter [m/s]')
        ax.set_ylabel('count (groups)')
        ax.set_title(f'A-B difference scatter ({ok.sum()}/{len(ab_rms)} groups < 5 km/s)')
        ax.legend(fontsize=8)

    # 4. Inter-chip gap scatter distribution
    ax = axes[1, 0]
    ok = chip_rms < 5000
    if ok.any():
        bins = np.linspace(0, 5000, 40)
        ax.hist(chip_rms[ok], bins=bins, color='C0', alpha=0.7, edgecolor='none')
        ax.axvline(np.median(chip_rms[ok]), color='k', ls='--', lw=1,
                   label=f'median={np.median(chip_rms[ok]):.0f} m/s')
        ax.set_xlabel('RMS scatter [m/s]')
        ax.set_ylabel('count (groups)')
        ax.set_title(f'Inter-chip gap scatter ({ok.sum()}/{len(chip_rms)} groups < 5 km/s)')
        ax.legend(fontsize=8)

    # 5. Inter-order gap scatter distribution
    ax = axes[1, 1]
    ok = order_rms < 5000
    if ok.any():
        bins = np.linspace(0, 5000, 40)
        ax.hist(order_rms[ok], bins=bins, color='C1', alpha=0.7, edgecolor='none')
        ax.axvline(np.median(order_rms[ok]), color='k', ls='--', lw=1,
                   label=f'median={np.median(order_rms[ok]):.0f} m/s')
        ax.set_xlabel('RMS scatter [m/s]')
        ax.set_ylabel('count (groups)')
        ax.set_title(f'Inter-order gap scatter ({ok.sum()}/{len(order_rms)} groups < 5 km/s)')
        ax.legend(fontsize=8)

    # 6. Combined comparison: CDF of all four metrics
    ax = axes[1, 2]
    for data, label, color in [
        (abs_rms_arr, 'absolute', 'C0'),
        (ab_rms, 'A-B', 'C2'),
        (chip_rms, 'inter-chip', 'C4'),
        (order_rms, 'inter-order', 'C1'),
    ]:
        ok = data < 5000
        if ok.any():
            sorted_d = np.sort(data[ok])
            cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
            ax.plot(sorted_d, cdf, label=label, color=color, lw=1.5)
    ax.set_xlabel('RMS scatter [m/s]')
    ax.set_ylabel('cumulative fraction')
    ax.set_title('CDF of per-group RMS (all metrics)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 5000)

    fig.tight_layout()
    fig.savefig('wavelength_precision.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nWrote wavelength_precision.png")


def main():
    reduced_dir = Path('reduced')
    prms_max = 10
    if len(sys.argv) > 1:
        prms_max = float(sys.argv[1])

    print(f"Collecting paired measurements (prms < {prms_max})...")
    paired = collect_paired(reduced_dir, prms_max=prms_max)

    print("Computing metrics...")
    abs_wl, chip_gaps, order_gaps, ab_diffs = compute_all_metrics(paired)

    abs_data = analyze_absolute(abs_wl)
    chip_rms, chip_counts = analyze_gaps(chip_gaps, "Inter-chip wavelength gaps")
    order_rms, order_counts = analyze_gaps(order_gaps, "Inter-order wavelength gaps")
    ab_rms, ab_counts = analyze_ab_diffs(ab_diffs)

    plot_results(abs_data, chip_rms, order_rms, ab_rms)


if __name__ == '__main__':
    main()
