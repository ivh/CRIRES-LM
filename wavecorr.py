#!/usr/bin/env python
"""Update wavelength scale in _tellcorr.fits using vipere solutions + 2D interpolation.

For orders with a vipere fit: uses the fitted wavelength polynomial directly.
For orders without: interpolates the wavelength *correction* (vipere - pipeline)
from a 2D polynomial fit delta_wl(x, order_number), fitted per chip.

Requires tellfit_{A,B}.par.dat and tellfit_{A,B}_xcen.json from tellcorr.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def get_coeffs(par, prefix):
    coeffs = []
    for i in range(10):
        k = f'{prefix}{i}'
        if k not in par:
            break
        coeffs.append(par[k])
    return coeffs


def vipere_order_to_chip_order(vorder, max_order):
    order_idx, det0 = divmod(vorder - 1, 3)
    chip = det0 + 1
    order_drs = max_order - order_idx
    return chip, order_drs


def fit_2d_wavelength(chip_data, deg_x=2, deg_ord=2):
    """Fit 2D polynomial to vipere wavelengths wl(x, order) on one chip.

    chip_data: list of (order_number, xcen, wave_coeffs)
    Returns function(x_array, order_number) -> wl in nm.
    """
    xs, ords, wls = [], [], []
    for odrs, xcen, wcoeffs in chip_data:
        xsamp = np.linspace(0, 2047, 20).astype(int)
        wl_vip = np.polynomial.polynomial.polyval(
            xsamp - xcen, wcoeffs) / 10.0
        xs.extend(xsamp.astype(float))
        ords.extend([float(odrs)] * len(xsamp))
        wls.extend(wl_vip)

    xs = np.array(xs)
    ords = np.array(ords)
    wls = np.array(wls)

    x_mean, x_std = xs.mean(), xs.std()
    o_mean, o_std = ords.mean(), max(ords.std(), 1.0)
    xn = (xs - x_mean) / x_std
    on = (ords - o_mean) / o_std

    powers = []
    for i in range(deg_x + 1):
        for j in range(deg_ord + 1):
            powers.append((i, j))
    A = np.column_stack([xn**i * on**j for i, j in powers])

    coeffs, _, _, _ = np.linalg.lstsq(A, wls, rcond=None)

    resid = wls - A @ coeffs
    mean_wl = np.mean(wls)
    rms_ms = np.std(resid) / mean_wl * 3e8
    print(f"    2D wavelength fit: {len(chip_data)} orders, "
          f"{len(xs)} points, rms={rms_ms:.0f} m/s")

    def eval_2d(x_arr, order_num):
        xn_ = (x_arr - x_mean) / x_std
        on_ = (order_num - o_mean) / o_std
        result = np.zeros_like(x_arr, dtype=float)
        for (i, j), c in zip(powers, coeffs):
            result += c * xn_**i * on_**j
        return result

    return eval_2d, (x_mean, x_std, o_mean, o_std, powers, coeffs)


def plot_2d_wavelength(chip_data, fit_info, eval_2d, chip,
                       orders_in_chip, unfitted_orders, outpath):
    """3D plot of wavelength surface and fitted data points."""
    x_mean, x_std, o_mean, o_std, powers, coeffs = fit_info

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    xline = np.linspace(0, 2047, 100)

    # data points
    for odrs, xcen, wcoeffs in chip_data:
        xsamp = np.linspace(0, 2047, 20).astype(int)
        wl_vip = np.polynomial.polynomial.polyval(
            xsamp - xcen, wcoeffs) / 10.0
        ax.scatter(xsamp, [odrs] * len(xsamp), wl_vip,
                   s=10, zorder=5)

    # lines from the surface for each order
    fitted_set = set(s[0] for s in chip_data)
    for odrs in orders_in_chip:
        wl_line = eval_2d(xline, odrs)
        if odrs in unfitted_orders:
            ax.plot(xline, [odrs] * len(xline), wl_line,
                    color='C3', lw=1.5, ls='--', zorder=4)
        else:
            ax.plot(xline, [odrs] * len(xline), wl_line,
                    color='C2', lw=1, alpha=0.7, zorder=4)

    # fitted surface
    all_orders = sorted(fitted_set)
    o_min = min(min(all_orders), min(orders_in_chip)) - 0.5
    o_max = max(max(all_orders), max(orders_in_chip)) + 0.5
    xg = np.linspace(0, 2047, 50)
    og = np.linspace(o_min, o_max, 50)
    XG, OG = np.meshgrid(xg, og)
    WL = eval_2d(XG.ravel(), OG.ravel()).reshape(XG.shape)
    ax.plot_surface(XG, OG, WL, alpha=0.2, color='C0')

    ax.plot([], [], color='C2', lw=1, label='fitted orders')
    ax.plot([], [], color='C3', lw=1.5, ls='--', label='interpolated orders')
    ax.legend(loc='upper left', fontsize=8)

    ax.view_init(elev=25, azim=-225)
    ax.set_xlabel('pixel')
    ax.set_ylabel('order')
    ax.set_zlabel('wavelength [nm]')
    ax.set_title(f'chip {chip} wavelength fit')

    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Wrote {outpath}")


def process_one(tellcorr_fits, pardat_file, xcen_file, ab='A'):
    """Update WL columns in a _tellcorr.fits."""
    hdul = fits.open(tellcorr_fits, mode='update')
    pars = parse_pardat(pardat_file)
    with open(xcen_file) as f:
        xcen_map = json.load(f)

    max_order = max(
        int(c.split('_')[0])
        for c in hdul[1].columns.names if c.endswith('_SPEC')
    )

    chip_solutions = {1: [], 2: [], 3: []}

    for par in pars:
        vorder = int(par.get('order', 0))
        if vorder == 0:
            continue
        prms = par.get('prms', -1)
        if prms < 0:
            continue

        chip, odrs = vipere_order_to_chip_order(vorder, max_order)
        wcoeffs = get_coeffs(par, 'wave')
        if not wcoeffs:
            continue

        xcen_key = f"{chip}_{odrs:02d}"
        if xcen_key not in xcen_map:
            continue
        xcen = xcen_map[xcen_key]

        chip_solutions[chip].append((odrs, xcen, wcoeffs))

    pixels = np.arange(2048, dtype=float)

    for chip in [1, 2, 3]:
        solutions = chip_solutions[chip]
        orders_in_chip = sorted(set(
            int(c.split('_')[0])
            for c in hdul[chip].columns.names if c.endswith('_SPEC')
        ))

        fitted_orders = set(s[0] for s in solutions)
        unfitted_orders = [o for o in orders_in_chip if o not in fitted_orders]

        print(f"  chip{chip}: fitted={sorted(fitted_orders)}, "
              f"unfitted={unfitted_orders}")

        if not solutions:
            print(f"    no solutions, skipping")
            continue

        # direct vipere WL for fitted orders
        for odrs, xcen, wcoeffs in solutions:
            wl_col = f"{odrs:02d}_01_WL"
            wl_vipere = np.polynomial.polynomial.polyval(
                pixels - xcen, wcoeffs) / 10.0
            hdul[chip].data[wl_col] = wl_vipere

        # 2D interpolation for unfitted orders
        if unfitted_orders and len(solutions) >= 2:
            eval_2d, fit_info = fit_2d_wavelength(solutions)
            plot_2d_wavelength(
                solutions, fit_info, eval_2d, chip,
                orders_in_chip, unfitted_orders,
                Path(tellcorr_fits).parent / f'wavecorr_chip{chip}_{ab}.png')
            for odrs in unfitted_orders:
                wl_col = f"{odrs:02d}_01_WL"
                wl_interp = eval_2d(pixels, odrs)
                hdul[chip].data[wl_col] = wl_interp
                print(f"    order {odrs:02d}: interpolated "
                      f"(center wl {wl_interp[1024]:.2f} nm)")
        elif unfitted_orders:
            print(f"    too few fitted orders for 2D fit, "
                  f"unfitted orders keep pipeline WL")

    hdul.flush()
    hdul.close()


def process_dir(dir_path):
    dir_path = Path(dir_path).resolve()

    for ab in ['A', 'B']:
        tellcorr = dir_path / f'cr2res_obs_nodding_extracted{ab}_tellcorr.fits'
        pardat = dir_path / f'tellfit_{ab}.par.dat'
        xcen = dir_path / f'tellfit_{ab}_xcen.json'

        if not tellcorr.exists():
            print(f"  Skipping {ab}: no tellcorr file")
            continue
        if not pardat.exists() or not xcen.exists():
            print(f"  Skipping {ab}: no par.dat/xcen (run tellcorr.py first)")
            continue

        print(f"\n--- Wavelength correction {ab} ---")
        process_one(tellcorr, pardat, xcen, ab=ab)
        print(f"  Updated {tellcorr}")


def main():
    parser = argparse.ArgumentParser(
        description='Update wavelength scale in tellcorr FITS using '
                    'vipere solutions + 2D interpolation')
    parser.add_argument('dir', help='Reduction directory')
    args = parser.parse_args()
    process_dir(args.dir)


if __name__ == '__main__':
    main()
