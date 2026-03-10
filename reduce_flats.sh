#!/bin/bash
set -euo pipefail

parallel -j32 --eta '
    sof={}
    setting=$(basename "$sof" | sed "s/_[0-9].*//")
    mkdir -p "$setting"
    esorex --output-dir="$setting" cr2res_cal_flat "$sof" \
        > "$setting/$(basename "$sof" .sof).log" 2>&1
' ::: sof/*_flats.sof
