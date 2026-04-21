#!/usr/bin/env python
# /// script
# dependencies = ["astropy", "matplotlib", "numpy"]
# ///
"""Plot order 7 (all 3 chips) from flat vs no-flat reduction -- uncorrected spectra + telluric model."""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

dir_flat = "reduced/bet_Pic_b_M4368_2024-09-19_08092"
dir_noflat = "reduced_noflat/bet_Pic_b_M4368_2024-09-19_08092"

fig, ax = plt.subplots(figsize=(7, 3.5))

offset_flat = 1.0
offset_noflat = 0.7

for chip in [1, 2, 3]:
    ext = f"CHIP{chip}.INT1"
    with fits.open(f"{dir_flat}/{dir_flat.split('/')[-1]}_tellcorrA.fits") as h:
        spec_flat = h[ext].data["07_01_SPEC"]
        wl = h[ext].data["07_01_WL"]
        tellur = h[ext].data["07_01_TELLUR"]
        cont = h[ext].data["07_01_CONT"]
    with fits.open(f"{dir_noflat}/{dir_noflat.split('/')[-1]}_tellcorrA.fits") as h:
        spec_nf = h[ext].data["07_01_SPEC"]
        tellur_nf = h[ext].data["07_01_TELLUR"]
        cont_nf = h[ext].data["07_01_CONT"]

    good = np.isfinite(spec_flat) & np.isfinite(wl) & np.isfinite(tellur) & np.isfinite(cont)
    good_nf = np.isfinite(spec_nf) & np.isfinite(wl) & np.isfinite(tellur_nf) & np.isfinite(cont_nf)

    # data / cont gives continuum-normalized corrected spectrum (~1 in continuum)
    # multiply by telluric to get uncorrected, normalized to continuum=1
    uncorr_flat = (spec_flat[good] / cont[good]) * tellur[good]
    uncorr_nf = (spec_nf[good_nf] / cont_nf[good_nf]) * tellur_nf[good_nf]

    # Normalize so continuum=1 using a high percentile (robust to absorption lines)
    med_flat = np.nanpercentile(uncorr_flat, 95)
    med_nf = np.nanpercentile(uncorr_nf, 95)

    lbl_nf = "Without flat" if chip == 1 else None
    lbl_flat = "With flat" if chip == 1 else None
    lbl_tell = "Telluric model" if chip == 1 else None

    ax.plot(wl[good_nf], uncorr_nf / med_nf -1+offset_noflat, color="C3", lw=0.4, alpha=0.9, label=lbl_nf)
    ax.plot(wl[good], uncorr_flat / med_flat -1+offset_flat, color="C0", lw=0.4, alpha=0.9, label=lbl_flat)
    ax.plot(wl[good_nf], tellur_nf[good_nf] / med_nf -1+offset_noflat, color="k", lw=0.3, alpha=0.8, label=lbl_tell)
    ax.plot(wl[good], tellur[good] / med_flat -1+offset_flat, color="k", lw=0.3, alpha=0.8)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Normalized flux")
fig.tight_layout()
fig.savefig("paper/figs/flat_comparison.pdf", dpi=150)
fig.savefig("paper/figs/flat_comparison.png", dpi=150)
print("Saved paper/figs/flat_comparison.{pdf,png}")
