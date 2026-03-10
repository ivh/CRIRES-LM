# /// script
# requires-python = ">=3.10"
# dependencies = ["astropy", "numpy", "matplotlib"]
# ///
"""Measure slit tilt from vipere A-B wavelength differences across all settings."""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent
PIXEL_SCALE = 0.057  # arcsec/pixel
N_SAMPLE = 10  # sample points per segment


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


def get_nod_throw(setting_dir):
    for ab in ["A", "B"]:
        f = setting_dir / f"cr2res_obs_nodding_extracted{ab}.fits"
        if f.exists():
            h = fits.getheader(f)
            return h.get("ESO SEQ NODTHROW", 6.0)
    return 6.0


# Collect tilt measurements: (setting, det, crires_order, sample_x[], tilt[])
all_tilts = []

settings = sorted(d.name for d in BASE.iterdir() if d.is_dir()
                  and (d / "telluricA.par.dat").exists()
                  and (d / "telluricB.par.dat").exists()
                  and (d / "telluricB.par.dat").stat().st_size > 100)

for setting in settings:
    sdir = BASE / setting
    parA = np.genfromtxt(sdir / "telluricA.par.dat", names=True, dtype=None, encoding=None)
    parB = np.genfromtxt(sdir / "telluricB.par.dat", names=True, dtype=None, encoding=None)
    if parA.ndim == 0: parA = parA.reshape(1)
    if parB.ndim == 0: parB = parB.reshape(1)

    hduA = fits.open(sdir / "cr2res_obs_nodding_extractedA.fits")
    hduB = fits.open(sdir / "cr2res_obs_nodding_extractedB.fits")

    nod_throw = get_nod_throw(sdir)
    delta_y_pix = nod_throw / PIXEL_SCALE

    dictA = {int(r["order"]): r for r in parA}
    dictB = {int(r["order"]): r for r in parB}
    common = sorted(set(dictA) & set(dictB))

    for oi in common:
        rowA, rowB = dictA[oi], dictB[oi]
        det, odr = vipere_order_to_crires(oi, hduA)
        ext = f"CHIP{det}.INT1"

        dataA = hduA[ext].data
        col_wl = f"{odr:02d}_01_WL"
        col_spec = f"{odr:02d}_01_SPEC"
        if col_wl not in dataA.names:
            continue

        wlA_orig = dataA[col_wl]
        specA = dataA[col_spec]
        wlB_orig = hduB[ext].data[col_wl]
        specB = hduB[ext].data[col_spec]

        goodA = np.isfinite(wlA_orig) & np.isfinite(specA) & (specA != 0) & (wlA_orig > 0)
        goodB = np.isfinite(wlB_orig) & np.isfinite(specB) & (specB != 0) & (wlB_orig > 0)
        good = goodA & goodB
        if good.sum() < 50:
            continue

        pixel = np.arange(len(wlA_orig))
        pixel_ok = pixel[good]

        wlA = reconstruct_wl(rowA, pixel_ok)  # Angstrom
        wlB = reconstruct_wl(rowB, pixel_ok)

        # Sample at N_SAMPLE points
        idx = np.linspace(0, len(pixel_ok) - 1, N_SAMPLE + 2, dtype=int)[1:-1]
        sample_pix = pixel_ok[idx]
        sample_wlA = wlA[idx]
        sample_wlB = wlB[idx]

        # Local dispersion (Angstrom/pixel)
        dwl_dx = np.gradient(wlA, pixel_ok)[idx]

        delta_wl = sample_wlA - sample_wlB
        # Tilt = spectral pixel shift / spatial pixel separation
        delta_x = delta_wl / dwl_dx
        tilt = delta_x / delta_y_pix

        # Quality check: reject if prms is bad or tilt is huge
        prmsA = rowA["prms"]
        prmsB = rowB["prms"]
        if prmsA > 20 or prmsB > 20:
            continue
        if np.any(np.abs(tilt) > 0.5):
            continue

        all_tilts.append((setting, det, odr, sample_pix, tilt, nod_throw))

    hduA.close()
    hduB.close()

print(f"Collected {len(all_tilts)} order segments across {len(settings)} settings")

# ---- Plot 1: Tilt vs pixel, grouped by detector, colored by CRIRES order ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
order_colors = plt.cm.tab10(np.linspace(0, 1, 10))

for setting, det, odr, pix, tilt, nod in all_tilts:
    ax = axes[det - 1]
    c = order_colors[odr % 10]
    ax.plot(pix, tilt, 'o-', color=c, ms=2, lw=0.5, alpha=0.5)

for det in range(3):
    axes[det].set_title(f"CHIP{det+1}")
    axes[det].set_xlabel("Pixel X")
    axes[det].axhline(0, color="k", ls=":", lw=0.5)
    axes[det].grid(True, alpha=0.3)
axes[0].set_ylabel("Slit tilt (dx/dy) [pix/pix]")

# Legend with CRIRES order numbers
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=order_colors[o % 10], marker='o', ms=3, lw=1)
           for o in range(2, 9)]
axes[2].legend(handles, [f"order {o}" for o in range(2, 9)], fontsize=7, loc="best")
fig.suptitle("Slit tilt measured from telluric A-B wavelength shift")
plt.tight_layout()
plt.savefig(str(BASE / "slit_tilt_by_chip.png"), dpi=150)

# ---- Plot 2: Same data but showing cross-detector continuity per spectral order ----
# For each setting, group segments by CRIRES spectral order across chips
fig2, axes2 = plt.subplots(len(settings), 1, figsize=(14, 3 * len(settings)), squeeze=False)

for si, setting in enumerate(settings):
    ax = axes2[si, 0]
    segs = [(s, det, odr, pix, tilt, nod) for s, det, odr, pix, tilt, nod in all_tilts if s == setting]

    # Group by CRIRES order
    from collections import defaultdict
    by_order = defaultdict(list)
    for s, det, odr, pix, tilt, nod in segs:
        by_order[odr].append((det, pix, tilt))

    for odr, chips in sorted(by_order.items()):
        c = order_colors[odr % 10]
        for det, pix, tilt in sorted(chips):
            # Offset pixel by chip: chip1 at 0-2047, chip2 at 2048-4095, chip3 at 4096-6143
            pix_global = pix + (det - 1) * 2048
            ax.plot(pix_global, tilt, 'o-', color=c, ms=2, lw=0.8, alpha=0.7)

    ax.set_title(f"{setting}")
    ax.set_ylabel("dx/dy")
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.grid(True, alpha=0.3)
    # Chip boundaries
    for xb in [2048, 4096]:
        ax.axvline(xb, color="gray", ls="--", lw=0.5, alpha=0.5)

axes2[-1, 0].set_xlabel("Global pixel (chip1 | chip2 | chip3)")
handles = [Line2D([0], [0], color=order_colors[o % 10], marker='o', ms=3, lw=1)
           for o in range(2, 9)]
axes2[0, 0].legend(handles, [f"order {o}" for o in range(2, 9)], fontsize=6, ncol=7, loc="upper right")
fig2.suptitle("Slit tilt cross-detector continuity per spectral order")
plt.tight_layout()
plt.savefig(str(BASE / "slit_tilt_continuity.png"), dpi=150)

# ---- Summary: median tilt per chip ----
for det in [1, 2, 3]:
    tilts = np.concatenate([tilt for s, d, o, p, tilt, n in all_tilts if d == det])
    print(f"CHIP{det}: median tilt = {np.median(tilts):.5f} dx/dy, MAD = {np.median(np.abs(tilts - np.median(tilts))):.5f}")

print("\nDone.")
