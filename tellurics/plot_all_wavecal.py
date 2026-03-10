# /// script
# requires-python = ">=3.10"
# dependencies = ["astropy", "numpy", "matplotlib"]
# ///
"""Compare vipere A vs B wavelength scales for all settings."""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent


def vipere_order_to_crires(order_idx, hdu):
    det = ((order_idx - 1) % 3) + 1
    spec_order = ((order_idx - 1) // 3)
    ext = f"CHIP{det}.INT1"
    data = hdu[ext].data
    n_orders = max(int(c.split('_')[0]) for c in data.names if c.endswith('_SPEC'))
    odr = n_orders - spec_order
    return ext, odr, det


def get_wavescales(par_file, fits_file):
    par = np.genfromtxt(par_file, names=True, dtype=None, encoding=None)
    if par.ndim == 0:
        par = par.reshape(1)
    hdu = fits.open(fits_file)
    results = []
    for row in par:
        order_idx = int(row["order"])
        ext, odr, det = vipere_order_to_crires(order_idx, hdu)
        data = hdu[ext].data
        col_wl = f"{odr:02d}_01_WL"
        col_spec = f"{odr:02d}_01_SPEC"
        if col_wl not in data.names:
            continue
        wl_orig = data[col_wl].copy()
        spec = data[col_spec].copy()
        good = np.isfinite(wl_orig) & np.isfinite(spec) & (spec != 0) & (wl_orig > 0)
        if good.sum() < 10:
            continue
        pixel = np.arange(len(wl_orig))
        pixel_ok = pixel[good]
        xcen = np.nanmean(pixel_ok) + 18
        wl_vipere = np.poly1d([row["wave2"], row["wave1"], row["wave0"]])(pixel_ok - xcen)
        wl_orig_A = wl_orig[good] * 10
        results.append((order_idx, det, odr, pixel_ok, wl_orig_A, wl_vipere))
    hdu.close()
    return results


settings = sorted(d.name for d in BASE.iterdir() if d.is_dir()
                  and (d / "telluricA.par.dat").exists()
                  and (d / "telluricB.par.dat").exists()
                  and (d / "telluricB.par.dat").stat().st_size > 100)

n = len(settings)
fig, axes = plt.subplots(n, 2, figsize=(16, 3 * n), squeeze=False)

for row_i, setting in enumerate(settings):
    sdir = BASE / setting
    try:
        scalesA = get_wavescales(sdir / "telluricA.par.dat", sdir / "cr2res_obs_nodding_extractedA.fits")
        scalesB = get_wavescales(sdir / "telluricB.par.dat", sdir / "cr2res_obs_nodding_extractedB.fits")
    except Exception as e:
        print(f"{setting}: {e}")
        continue

    dictA = {s[0]: s for s in scalesA}
    dictB = {s[0]: s for s in scalesB}
    common = sorted(set(dictA) & set(dictB))
    colors = plt.cm.viridis(np.linspace(0, 1, max(dictA.keys() | dictB.keys())))

    ax_corr = axes[row_i, 0]
    ax_ab = axes[row_i, 1]

    for oi in common:
        _, detA, odrA, pixA, origA, vipA = dictA[oi]
        _, detB, odrB, pixB, origB, vipB = dictB[oi]
        c = colors[oi - 1]
        label = f"o{oi} (d{detA},ord{odrA})"

        corrA_kms = (vipA - origA) / origA * 3e5
        corrB_kms = (vipB - origB) / origB * 3e5
        ax_corr.plot(origA / 10, corrA_kms, color=c, lw=0.5, alpha=0.7)
        ax_corr.plot(origB / 10, corrB_kms, color=c, lw=0.5, alpha=0.7, ls="--")

        common_pix = np.intersect1d(pixA, pixB)
        if len(common_pix) < 10:
            continue
        idxA = np.searchsorted(pixA, common_pix)
        idxB = np.searchsorted(pixB, common_pix)
        diff_AB_kms = (vipA[idxA] - vipB[idxB]) / vipA[idxA] * 3e5
        ax_ab.plot(common_pix, diff_AB_kms, color=c, lw=0.8, alpha=0.7, label=label)

    ax_corr.set_title(f"{setting}: correction (solid=A, dashed=B)")
    ax_corr.set_ylabel("km/s")
    ax_corr.axhline(0, color="k", ls=":", lw=0.5)
    ax_ab.set_title(f"{setting}: A - B")
    ax_ab.set_ylabel("km/s")
    ax_ab.axhline(0, color="k", ls=":", lw=0.5)
    ax_ab.legend(fontsize=5, ncol=4, loc="best")

axes[-1, 0].set_xlabel("Wavelength [nm]")
axes[-1, 1].set_xlabel("Pixel")

plt.tight_layout()
plt.savefig(str(BASE / "wavecal_all.png"), dpi=150)
print(f"Saved wavecal_all.png with {n} settings")
