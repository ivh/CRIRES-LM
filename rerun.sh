#!/bin/bash
# Re-reduce an observation directory: adjust traces, extract, telluric correct, wavecal.
# Usage: ./rerun.sh reduced/target_setting_date_N [reduced/another_dir ...]

set -euo pipefail

for dir in "$@"; do
    dir="${dir%/}"
    echo "=== $dir ==="
    uv run adjust_traces.py "$dir"
    (cd "$dir" && esorex --recipe-config=../../cr2res_obs_nodding.rc cr2res_obs_nodding nodd.sof 2>&1 > esorex.log)
    uv run tellcorr.py "$dir"
    uv run wavecorr.py "$dir"
    uv run plot_ABtraces.py "$dir"
done
