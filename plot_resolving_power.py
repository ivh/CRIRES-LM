# /// script
# dependencies = ["astropy", "matplotlib", "numpy"]
# ///
"""Measure and plot spectral resolving power for all CRIRES+ L/M reductions.

R = lambda / (SLITFWHM * dispersion), where SLITFWHM is the pipeline-measured
slit illumination FWHM in pixels (assumed symmetric into dispersion direction).
"""

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def measure_R(filepath):
    """Return median R from one extractedA/B file, plus metadata."""
    with fits.open(filepath) as hdul:
        slit_wid = hdul[0].header.get("ESO INS SLIT1 WID")
        setting = hdul[0].header.get("ESO INS WLEN ID", "")

        Rs = []
        for chip in [1, 2, 3]:
            try:
                ext = hdul[f"CHIP{chip}.INT1"]
                if not hasattr(ext, "columns"):
                    continue
            except KeyError:
                continue
            for order in range(1, 12):
                col_wl = f"{order:02d}_01_WL"
                key_fwhm = f"ESO QC SLITFWHM{order}"
                if col_wl not in ext.columns.names or key_fwhm not in ext.header:
                    continue
                wl = ext.data[col_wl]
                good = wl > 0
                if good.sum() < 100:
                    continue
                wl_good = wl[good]
                wl_cen = wl_good[len(wl_good) // 2]
                disp = np.median(np.abs(np.diff(wl_good)))
                fwhm = ext.header[key_fwhm]
                if fwhm <= 0 or disp <= 0:
                    continue
                Rs.append(wl_cen / (fwhm * disp))

        if not Rs:
            return None
        return {
            "R_median": np.median(Rs),
            "slit_wid": slit_wid,
            "setting": setting,
        }


def main():
    files = sorted(
        glob.glob("reduced/*/cr2res_obs_nodding_extractedA.fits")
        + glob.glob("reduced/*/cr2res_obs_nodding_extractedB.fits")
    )
    print(f"Found {len(files)} extracted spectra")

    results = []
    for i, f in enumerate(files):
        if i % 500 == 0:
            print(f"  {i}/{len(files)}...", file=sys.stderr)
        r = measure_R(f)
        if r:
            results.append(r)

    print(f"Measured R for {len(results)} files")

    R_all = np.array([r["R_median"] for r in results])
    slits = np.array([r["slit_wid"] for r in results])
    settings = np.array([r["setting"] for r in results])

    slit_values = sorted(set(slits))
    is_L = np.array([s.startswith("L") for s in settings])

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for ax, slit in zip(axes, slit_values):
        mask = slits == slit
        R_slit = R_all[mask] / 1000
        L_mask = is_L[mask]

        bins = np.arange(20, 200, 3)
        n_L = L_mask.sum()
        n_M = (~L_mask).sum()
        ax.hist(R_slit[L_mask], bins=bins, alpha=0.7, label=f"L band (n={n_L})")
        ax.hist(R_slit[~L_mask], bins=bins, alpha=0.7, label=f"M band (n={n_M})")
        med = np.median(R_slit)
        ax.axvline(med, color="k", ls="--", lw=1, label=f"median {med:.0f}k")
        ax.set_ylabel("count")
        ax.set_title(f'Slit {slit}"')
        ax.legend()

    axes[-1].set_xlabel("Resolving power R / 1000")
    fig.suptitle("CRIRES+ L/M spectral resolving power (from slit FWHM)")
    fig.tight_layout()
    fig.savefig("resolving_power.png", dpi=150)
    print("Saved resolving_power.png")
    plt.show()


if __name__ == "__main__":
    main()
