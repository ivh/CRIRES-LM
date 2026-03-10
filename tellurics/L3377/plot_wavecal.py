# /// script
# requires-python = ">=3.10"
# dependencies = ["astropy", "numpy", "matplotlib"]
# ///
"""Compare vipere-fitted wavelength scales for extractedA vs extractedB in L3377."""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


def vipere_order_to_crires(order_idx, hdu):
    """Map vipere order index to (extension, CRIRES order number)."""
    det = ((order_idx - 1) % 3) + 1
    spec_order = ((order_idx - 1) // 3)
    ext = f"CHIP{det}.INT1"
    data = hdu[ext].data
    n_orders = max(int(c.split('_')[0]) for c in data.names if c.endswith('_SPEC'))
    odr = n_orders - spec_order
    return ext, odr


def get_wavescales(par_file, fits_file):
    """Return list of (order_idx, det, odr, pixel_ok, wl_orig_A, wl_vipere_A) per order."""
    par = np.genfromtxt(par_file, names=True, dtype=None, encoding=None)
    hdu = fits.open(fits_file)
    results = []
    for row in par:
        order_idx = int(row["order"])
        ext, odr = vipere_order_to_crires(order_idx, hdu)
        det = ((order_idx - 1) % 3) + 1
        data = hdu[ext].data
        wl_orig = data[f"{odr:02d}_01_WL"].copy()
        spec = data[f"{odr:02d}_01_SPEC"].copy()
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


scalesA = get_wavescales("telluricA.par.dat", "cr2res_obs_nodding_extractedA.fits")
scalesB = get_wavescales("telluric.par.dat", "cr2res_obs_nodding_extractedB.fits")

# Index by order_idx for easy pairing
dictA = {s[0]: s for s in scalesA}
dictB = {s[0]: s for s in scalesB}
common = sorted(set(dictA) & set(dictB))

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
ax_corr, ax_ab, ax_abpix = axes
colors = plt.cm.viridis(np.linspace(0, 1, len(common)))

for ci, oi in enumerate(common):
    oiA, detA, odrA, pixA, origA, vipA = dictA[oi]
    oiB, detB, odrB, pixB, origB, vipB = dictB[oi]
    c = colors[ci]
    label = f"o{oi} (d{detA},ord{odrA})"

    # Panel 1: correction vs original (both A and B, relative to their own pipeline WL)
    corrA_kms = (vipA - origA) / origA * 3e5
    corrB_kms = (vipB - origB) / origB * 3e5
    ax_corr.plot(origA / 10, corrA_kms, color=c, lw=0.5, alpha=0.7, label=label)
    ax_corr.plot(origB / 10, corrB_kms, color=c, lw=0.5, alpha=0.7, ls="--")

    # Panel 2: A-B vipere wavelength difference vs wavelength
    # Interpolate B onto A's pixel grid for comparison
    common_pix = np.intersect1d(pixA, pixB)
    if len(common_pix) < 10:
        continue
    idxA = np.searchsorted(pixA, common_pix)
    idxB = np.searchsorted(pixB, common_pix)
    diff_AB_kms = (vipA[idxA] - vipB[idxB]) / vipA[idxA] * 3e5
    wl_nm = vipA[idxA] / 10
    ax_ab.plot(wl_nm, diff_AB_kms, color=c, lw=0.8, alpha=0.7, label=label)
    ax_abpix.plot(common_pix, diff_AB_kms, color=c, lw=0.8, alpha=0.7, label=label)

ax_corr.set_xlabel("Pipeline wavelength [nm]")
ax_corr.set_ylabel("Correction [km/s]")
ax_corr.set_title("L3377: Vipere correction vs pipeline WL (solid=A, dashed=B)")
ax_corr.legend(fontsize=6, ncol=4, loc="best")
ax_corr.axhline(0, color="k", ls=":", lw=0.5)

ax_ab.set_xlabel("Wavelength [nm]")
ax_ab.set_ylabel("A - B [km/s]")
ax_ab.set_title("Vipere wavelength difference: extractedA - extractedB")
ax_ab.legend(fontsize=6, ncol=4, loc="best")
ax_ab.axhline(0, color="k", ls=":", lw=0.5)

ax_abpix.set_xlabel("Pixel")
ax_abpix.set_ylabel("A - B [km/s]")
ax_abpix.set_title("Vipere wavelength difference vs pixel: extractedA - extractedB")
ax_abpix.axhline(0, color="k", ls=":", lw=0.5)

plt.tight_layout()
plt.savefig("wavecal_comparison.png", dpi=150)
plt.show()
print("Saved wavecal_comparison.png")
