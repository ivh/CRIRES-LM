# /// script
# requires-python = ">=3.10"
# dependencies = ["astropy", "scipy", "numpy"]
# ///
"""Measure trace Y-shift from raw science frames and write adjusted _tw.fits.

For each reduction directory, cross-correlates the spatial edge profile of a
raw frame against the order boundaries in the _tw.fits to find the Y offset
due to instrument flexure. Writes an adjusted copy of _tw.fits into the
reduction directory and updates the SOF to use it.

Usage:
    uv run adjust_traces.py reduced/some_dir
    uv run adjust_traces.py reduced/some_dir reduced/other_dir ...
    ls reduced/*/nodd.sof | sed 's|/nodd.sof||' | parallel -j8 --bar 'uv run adjust_traces.py {}'
"""

import sys
import numpy as np
from pathlib import Path
from astropy.io import fits
from scipy.ndimage import uniform_filter1d

MAX_SHIFT = 100


def measure_shift(raw_path, tw_path):
    """Return per-chip Y-shifts (pixels) by edge cross-correlation."""
    raw = fits.open(raw_path)
    tw = fits.open(tw_path)
    shifts = {}

    for chip_idx in range(3):
        chip_name = f"CHIP{chip_idx+1}.INT1"
        data = raw[chip_name].data
        tw_tab = tw[chip_name].data

        # spatial profile: median across central half of detector
        observed = np.nanmedian(data[:, 500:1500], axis=1).astype(float)
        obs_smooth = uniform_filter1d(observed, 5)
        obs_edges = np.abs(np.gradient(obs_smooth))
        obs_edges[:50] = 0
        obs_edges[2000:] = 0

        # synthetic edge profile from trace boundaries at detector center
        xcol = 1024
        syn_edges = np.zeros(2048)
        for row in tw_tab:
            y_upper = np.polyval(row["Upper"][::-1], xcol)
            y_lower = np.polyval(row["Lower"][::-1], xcol)
            for ye in [y_lower, y_upper]:
                yi = int(round(ye))
                for delta in range(-5, 6):
                    idx = yi + delta
                    if 0 <= idx < 2048:
                        syn_edges[idx] += np.exp(-0.5 * (delta / 2.0) ** 2)

        # cross-correlate
        best_cc = -np.inf
        best_dy = 0
        cc_at = {}
        for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
            shifted = np.roll(syn_edges, dy)
            cc = np.sum(obs_edges * shifted)
            cc_at[dy] = cc
            if cc > best_cc:
                best_cc = cc
                best_dy = dy

        # sub-pixel parabolic refinement
        if -MAX_SHIFT < best_dy < MAX_SHIFT:
            y0, y1, y2 = cc_at[best_dy - 1], cc_at[best_dy], cc_at[best_dy + 1]
            denom = y0 - 2 * y1 + y2
            subpix = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-10 else 0
            shift_fine = best_dy + subpix
        else:
            shift_fine = float(best_dy)

        shifts[chip_idx + 1] = shift_fine

    raw.close()
    tw.close()
    return shifts


def apply_shift(tw_path, out_path, shifts):
    """Copy tw_path to out_path, adding per-chip Y-shift to trace polynomials."""
    tw = fits.open(tw_path)

    for chip_idx in range(3):
        chip_name = f"CHIP{chip_idx+1}.INT1"
        dy = shifts[chip_idx + 1]
        tw_tab = tw[chip_name].data
        for row in tw_tab:
            for col in ["All", "Upper", "Lower"]:
                row[col][0] += dy

    tw.writeto(out_path, overwrite=True)
    tw.close()


def process_dir(dirpath):
    dirpath = Path(dirpath)
    sof_path = dirpath / "nodd.sof"
    if not sof_path.exists():
        print(f"SKIP {dirpath}: no nodd.sof")
        return

    sof_text = sof_path.read_text()

    # find TW filename from SOF (e.g. ./L3377_tw.fits)
    tw_line = None
    for line in sof_text.splitlines():
        if "UTIL_WAVE_TW" in line:
            tw_line = line.strip()
            break
    if tw_line is None:
        print(f"SKIP {dirpath}: no UTIL_WAVE_TW in SOF")
        return

    tw_name = Path(tw_line.split()[0]).name  # e.g. L3377_tw.fits
    # original (unadjusted) _tw.fits lives two levels up
    tw_orig = dirpath / ".." / ".." / tw_name
    if not tw_orig.exists():
        print(f"SKIP {dirpath}: {tw_orig} not found")
        return

    # pick first raw frame from SOF
    raw_ref = None
    for line in sof_text.splitlines():
        if "OBS_NODDING_OTHER" in line:
            raw_ref = line.split()[0]
            break
    if raw_ref is None:
        print(f"SKIP {dirpath}: no raw frames in SOF")
        return
    raw_path = dirpath / raw_ref

    shifts = measure_shift(str(raw_path), str(tw_orig))
    median_shift = np.median(list(shifts.values()))
    shifts_applied = {c: median_shift for c in [1, 2, 3]}

    local_tw = dirpath / tw_name
    apply_shift(str(tw_orig), str(local_tw), shifts_applied)

    print(f"{dirpath.name}: shift={median_shift:+.1f} px  chips=[{shifts[1]:+.1f}, {shifts[2]:+.1f}, {shifts[3]:+.1f}]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for d in sys.argv[1:]:
        process_dir(d)
