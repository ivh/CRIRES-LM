#!/usr/bin/env python3
"""Measure actual nod throw from combinedA frames by fitting Gaussians to A/B traces."""

import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from pathlib import Path
import json

SETTINGS = {
    'L3244': 6, 'L3262': 6, 'L3302': 6,
    'L3377': 7, 'L3412': 7, 'L3426': 7,
    'M4187': 6, 'M4211': 5, 'M4266': 7,
    'M4318': 6, 'M4368': 6, 'M4416': 6,
    'M4461': 5, 'M4504': 5,
}

PIXEL_SCALE = 0.059  # arcsec/pixel (approximate, from slit length / trace width)

def double_gaussian(y, a1, mu1, sig1, a2, mu2, sig2, bg):
    return (a1 * np.exp(-0.5 * ((y - mu1) / sig1)**2) +
            a2 * np.exp(-0.5 * ((y - mu2) / sig2)**2) + bg)


def measure_setting(setting, n_orders, base_dir=Path('.')):
    setting_dir = base_dir / setting
    combined = setting_dir / 'cr2res_obs_nodding_combinedA.fits'
    tw_file = base_dir.parent / f'{setting}_tw.fits'

    if not combined.exists() or not tw_file.exists():
        print(f'{setting}: missing files, skipping')
        return []

    hdul = fits.open(combined)
    tw = fits.open(tw_file)

    # read nod throw from header
    nod_throw = hdul[0].header.get('ESO SEQ NODTHROW', None)

    results = []
    xcen = 1024
    hw = 10

    for chip in [1, 2, 3]:
        ext = f'CHIP{chip}.INT1'
        data = hdul[ext].data
        tab = tw[ext].data

        cut = np.nanmedian(data[:, xcen-hw:xcen+hw], axis=1)

        for row in tab:
            odr = row['Order']
            y_center = sum(c * xcen**j for j, c in enumerate(row['All']))
            y_upper = sum(c * xcen**j for j, c in enumerate(row['Upper']))
            y_lower = sum(c * xcen**j for j, c in enumerate(row['Lower']))
            trace_hw = (y_upper - y_lower) / 2

            # expected nod positions: center +/- throw/2/pixscale
            # but we don't know pixscale precisely, so use the data
            # search region: from lower-10 to upper+10
            y_lo = max(0, int(y_lower - 10))
            y_hi = min(2047, int(y_upper + 10))
            yy = np.arange(y_lo, y_hi + 1)
            profile = np.abs(cut[y_lo:y_hi+1])

            if np.all(np.isnan(profile)) or np.nanmax(profile) < 10:
                continue

            # initial guesses: two peaks symmetric around center
            half_throw_pix = nod_throw / 2 / PIXEL_SCALE if nod_throw else trace_hw / 2
            mu1_guess = y_center - half_throw_pix / 2
            mu2_guess = y_center + half_throw_pix / 2

            # refine: find the two highest peaks in the profile
            from scipy.signal import find_peaks
            peaks, props = find_peaks(profile, height=np.nanmax(profile) * 0.1,
                                      distance=10)
            if len(peaks) >= 2:
                # take the two tallest
                idx = np.argsort(props['peak_heights'])[-2:]
                pk = sorted(peaks[idx])
                mu1_guess = yy[pk[0]]
                mu2_guess = yy[pk[1]]

            amp_guess = np.nanmax(profile)
            sig_guess = 3.0

            p0 = [amp_guess, mu1_guess, sig_guess,
                  amp_guess, mu2_guess, sig_guess, 0]
            bounds_lo = [0, yy[0], 0.5, 0, yy[0], 0.5, -amp_guess]
            bounds_hi = [amp_guess*5, yy[-1], 30, amp_guess*5, yy[-1], 30, amp_guess]

            try:
                good = ~np.isnan(profile)
                popt, pcov = curve_fit(double_gaussian, yy[good], profile[good],
                                       p0=p0, bounds=(bounds_lo, bounds_hi),
                                       maxfev=10000)
                a1, mu1, sig1, a2, mu2, sig2, bg = popt
                perr = np.sqrt(np.diag(pcov))

                # ensure mu1 < mu2
                if mu1 > mu2:
                    mu1, mu2 = mu2, mu1
                    sig1, sig2 = sig2, sig1

                dy = mu2 - mu1
                dy_err = np.sqrt(perr[1]**2 + perr[4]**2)
                actual_throw = dy * PIXEL_SCALE

                results.append({
                    'setting': setting,
                    'chip': chip,
                    'order': int(odr),
                    'mu_A': float(mu1),
                    'mu_B': float(mu2),
                    'sig_A': float(sig1),
                    'sig_B': float(sig2),
                    'dy_pix': float(dy),
                    'dy_err': float(dy_err),
                    'nod_throw_cmd': float(nod_throw) if nod_throw else None,
                    'nod_throw_meas': float(actual_throw),
                })
            except Exception as e:
                print(f'  {setting} CHIP{chip} O{odr}: fit failed ({e})')

    hdul.close()
    tw.close()
    return results


def main():
    base_dir = Path(__file__).parent
    all_results = []

    for setting, n_orders in sorted(SETTINGS.items()):
        print(f'{setting} ({n_orders} orders)...')
        res = measure_setting(setting, n_orders, base_dir)
        all_results.extend(res)
        for r in res:
            print(f"  CHIP{r['chip']} O{r['order']}: dy={r['dy_pix']:.2f} +/- {r['dy_err']:.3f} pix"
                  f"  ({r['nod_throw_meas']:.3f}\" vs cmd {r['nod_throw_cmd']}\")")

    # quality filter: reject fits with large errors or dy far from expected
    # first pass: rough cut
    good_results = []
    for r in all_results:
        expected_dy = r['nod_throw_cmd'] / PIXEL_SCALE if r['nod_throw_cmd'] else 100
        if r['dy_err'] < 0.2 and abs(r['dy_pix'] - expected_dy) < 10:
            good_results.append(r)

    # second pass: per-setting sigma clip
    filtered = []
    for setting in sorted(SETTINGS):
        subset = [r for r in good_results if r['setting'] == setting]
        if len(subset) < 3:
            continue
        dy_med = np.median([r['dy_pix'] for r in subset])
        dy_mad = np.median([abs(r['dy_pix'] - dy_med) for r in subset])
        for r in subset:
            if abs(r['dy_pix'] - dy_med) < max(5 * dy_mad, 1.0):
                filtered.append(r)
    good_results = filtered
    print(f'\n=== Quality filter: {len(good_results)}/{len(all_results)} passed ===')

    # summary
    for throw_cmd in sorted(set(r['nod_throw_cmd'] for r in good_results if r['nod_throw_cmd'])):
        subset = [r for r in good_results if r['nod_throw_cmd'] == throw_cmd]
        dy_vals = np.array([r['dy_pix'] for r in subset])
        throws = np.array([r['nod_throw_meas'] for r in subset])
        print(f'Commanded {throw_cmd}":  dy = {np.median(dy_vals):.2f} +/- {np.std(dy_vals):.2f} pix'
              f'  =  {np.median(throws):.3f} +/- {np.std(throws):.3f}"')

    # per-chip summary
    for chip in [1, 2, 3]:
        subset = [r for r in good_results if r['chip'] == chip]
        if subset:
            dy_vals = np.array([r['dy_pix'] for r in subset])
            print(f'CHIP{chip}: median dy = {np.median(dy_vals):.2f} pix')

    # save results
    outfile = base_dir / 'nod_throw_measurements.json'
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved {len(all_results)} measurements to {outfile}')

    # per-setting summary
    print(f'\n{"setting":>8s}  {"cmd":>5s}  {"N":>3s}  {"dy_med":>7s}  {"dy_std":>7s}  {"meas":>7s}  {"ratio":>6s}')
    for setting in sorted(SETTINGS):
        subset = [r for r in good_results if r['setting'] == setting]
        if not subset:
            continue
        dy = np.array([r['dy_pix'] for r in subset])
        cmd = subset[0]['nod_throw_cmd']
        meas = np.median(dy) * PIXEL_SCALE
        print(f'{setting:>8s}  {cmd:5.1f}  {len(subset):3d}  {np.median(dy):7.2f}  {np.std(dy):7.3f}'
              f'  {meas:7.3f}  {meas/cmd:6.4f}')

    # plot: dy vs Y position (trace center), colored by commanded throw
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for throw_cmd, marker, color_base in [(5.5, 's', 'C1'), (6.0, 'o', 'C0')]:
        for chip, color in [(1, 'C0'), (2, 'C1'), (3, 'C2')]:
            subset = [r for r in good_results
                      if r['nod_throw_cmd'] == throw_cmd and r['chip'] == chip]
            if not subset:
                continue
            y_cen = [(r['mu_A'] + r['mu_B']) / 2 for r in subset]
            dy = [r['dy_pix'] for r in subset]
            label = f'cmd {throw_cmd}" CHIP{chip}' if chip == 2 else f'_CHIP{chip}'
            ax1.scatter(y_cen, dy, c=color, marker=marker, s=15, alpha=0.7,
                       label=label if chip == 2 else None)

    ax1.set_ylabel('dy (pix)')
    ax1.legend(fontsize=8)
    ax1.set_title('Nod throw: measured dy vs Y position on detector')

    # plot ratio: measured / commanded
    for chip, color in [(1, 'C0'), (2, 'C1'), (3, 'C2')]:
        subset = [r for r in good_results if r['chip'] == chip]
        y_cen = [(r['mu_A'] + r['mu_B']) / 2 for r in subset]
        ratio = [r['nod_throw_meas'] / r['nod_throw_cmd'] for r in subset]
        ax2.scatter(y_cen, ratio, c=color, s=15, alpha=0.7,
                   label=f'CHIP{chip}')

    ax2.set_xlabel('Y center (pix)')
    ax2.set_ylabel('measured / commanded')
    ax2.axhline(1.0, color='gray', ls='--', alpha=0.5)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(base_dir / 'nod_throw_vs_y.png', dpi=150)
    print(f'Saved nod_throw_vs_y.png')


if __name__ == '__main__':
    main()
