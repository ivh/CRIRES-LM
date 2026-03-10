# /// script
# requires-python = ">=3.10"
# dependencies = ["astropy", "numpy", "matplotlib"]
# ///
"""Plot A-B wavelength difference and slit tilt vs wavelength for all settings."""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent
PIXEL_SCALE = 0.057
N_SAMPLE = 10


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
            return fits.getheader(f).get("ESO SEQ NODTHROW", 6.0)
    return 6.0


settings = sorted(d.name for d in BASE.iterdir() if d.is_dir()
                  and (d / "telluricA.par.dat").exists()
                  and (d / "telluricB.par.dat").exists()
                  and (d / "telluricB.par.dat").stat().st_size > 100)

chip_markers = {1: 'o', 2: 's', 3: 'D'}
chip_colors = {1: 'C0', 2: 'C1', 3: 'C2'}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

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

    for oi in sorted(set(dictA) & set(dictB)):
        rowA, rowB = dictA[oi], dictB[oi]
        det, odr = vipere_order_to_crires(oi, hduA)
        ext = f"CHIP{det}.INT1"

        dataA = hduA[ext].data
        col_wl = f"{odr:02d}_01_WL"
        col_spec = f"{odr:02d}_01_SPEC"
        if col_wl not in dataA.names:
            continue

        specA = dataA[col_spec]
        specB = hduB[ext].data[col_spec]
        wlA_orig = dataA[col_wl]
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
        sample_wlA = wlA[idx]
        sample_wlB = wlB[idx]
        wl_nm = sample_wlA / 10

        delta_wl = sample_wlA - sample_wlB
        delta_wl_kms = delta_wl / sample_wlA * 3e5

        dwl_dx = np.gradient(wlA, pixel_ok)[idx]
        delta_x = delta_wl / dwl_dx
        tilt = delta_x / delta_y_pix

        prmsA, prmsB = rowA["prms"], rowB["prms"]
        if prmsA > 20 or prmsB > 20:
            continue
        if np.any(np.abs(tilt) > 0.5):
            continue

        c = chip_colors[det]
        m = chip_markers[det]
        ax1.plot(wl_nm, delta_wl_kms, marker=m, color=c, ms=2, lw=0.5, alpha=0.4)
        ax2.plot(wl_nm, tilt, marker=m, color=c, ms=2, lw=0.5, alpha=0.4)

    hduA.close()
    hduB.close()

ax1.set_ylabel("A - B [km/s]")
ax1.set_title("Wavelength difference (vipere A - B)")
ax1.axhline(0, color="k", ls=":", lw=0.5)
ax1.grid(True, alpha=0.3)

ax2.set_ylabel("Slit tilt dx/dy [pix/pix]")
ax2.set_xlabel("Wavelength [nm]")
ax2.set_title("Slit tilt")
ax2.axhline(0, color="k", ls=":", lw=0.5)
ax2.grid(True, alpha=0.3)

from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=chip_colors[d], marker=chip_markers[d], ms=5, lw=0)
           for d in [1, 2, 3]]
ax2.legend(handles, ["CHIP1", "CHIP2", "CHIP3"], loc="best")

plt.tight_layout()
plt.savefig(str(BASE / "tilt_vs_wl.png"), dpi=150)
print("Saved tilt_vs_wl.png")
