#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "astropy"]
# ///
"""Generate SOF files from FITS headers in raw/, without touching LMscience.sqlite.

Sibling of make_sofs.py for private/non-public data: reads the metadata it needs
straight out of the headers of every *.fits in raw/, then emits the same
combined-per-template and per-AB-pair directories under reduced/.
"""

import bisect
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from astropy.io import fits

BASE = Path(__file__).parent
RAW = BASE / 'raw'

TPL_NAME_KEEP = 'Nodding along slit using jitter'

rows = []
for fitsfile in sorted(RAW.glob('*.fits')):
    h = fits.getheader(fitsfile)
    rows.append({
        'dp_id': fitsfile.stem,
        'object': h.get('OBJECT', ''),
        'ins_wlen_id': h.get('HIERARCH ESO INS WLEN ID', ''),
        'tpl_start': h.get('HIERARCH ESO TPL START', ''),
        'tpl_name': h.get('HIERARCH ESO TPL NAME', ''),
        'nodpos': h.get('HIERARCH ESO SEQ NODPOS', ''),
        'date_obs': h.get('DATE-OBS', ''),
    })

df = pd.DataFrame(rows)
print(f'{len(df)} raw frames in raw/')

df = df[(df['nodpos'].isin(['A', 'B'])) & (df['tpl_name'] == TPL_NAME_KEEP)]
df = df.sort_values(['tpl_start', 'date_obs']).reset_index(drop=True)
print(f'{len(df)} frames with nod positions and matching template')

# build lookup of flat dates per setting from flats/ directory structure
flat_dates = {}  # setting -> sorted list of (datetime, dirname)
for d in sorted((BASE / 'flats').iterdir()):
    if not d.is_dir() or d.name in ('raw',):
        continue
    parts = d.name.rsplit('_', 1)
    if len(parts) != 2:
        continue
    setting, date_str = parts
    dt = datetime.fromisoformat(date_str)
    flat_dates.setdefault(setting, []).append((dt, d.name))


def nearest_flat(setting, obs_date):
    """Find the flat directory nearest in time to obs_date."""
    candidates = flat_dates.get(setting, [])
    if not candidates:
        return None
    dt = datetime.fromisoformat(obs_date[:10])
    idx = bisect.bisect_left(candidates, (dt,))
    best = None
    for i in (idx - 1, idx):
        if 0 <= i < len(candidates):
            if best is None or abs(candidates[i][0] - dt) < abs(best[0] - dt):
                best = candidates[i]
    return best[1]


def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9._+-]', '_', name)


def write_sofs(directory, raw_lines, tw, flat, blaze):
    """Write nodd.sof (raw + tw) and calib.sof (flat + blaze) into directory."""
    directory.mkdir(exist_ok=True)
    with open(directory / 'nodd.sof', 'w') as f:
        for line in raw_lines:
            f.write(line)
        f.write(f'{tw} UTIL_WAVE_TW\n')
    with open(directory / 'calib.sof', 'w') as f:
        f.write(f'{flat} CAL_FLAT_MASTER\n')
        f.write(f'{blaze} CAL_FLAT_EXTRACT_1D\n')


outdir = BASE / 'reduced'
outdir.mkdir(exist_ok=True)

n_pairs = 0
n_combined = 0
n_skipped = 0

for tpl_start, group in df.groupby('tpl_start'):
    setting = group.iloc[0]['ins_wlen_id']
    obj = group.iloc[0]['object']

    tw = f'./{setting}_tw.fits'
    flat_dir = nearest_flat(setting, tpl_start)
    if flat_dir is None:
        print(f'no flat for {setting} near {tpl_start}, skipping {len(group)} frames')
        n_skipped += len(group)
        continue
    flat = f'../../flats/{flat_dir}/cr2res_cal_flat_Open_master_flat.fits'
    blaze = f'../../flats/{flat_dir}/cr2res_cal_flat_Open_blaze.fits'
    tpl_short = tpl_start.replace('T', '_').replace(':', '')[:16]

    frames = group.sort_values('date_obs').reset_index(drop=True)

    # --- combined: all frames per template ---
    na = (frames['nodpos'] == 'A').sum()
    nb = (frames['nodpos'] == 'B').sum()
    comb_frames = frames
    if na != nb:
        n_keep = min(na, nb)
        if n_keep == 0:
            n_skipped += len(frames)
            continue
        comb_frames = pd.concat([
            frames[frames['nodpos'] == 'A'].iloc[:n_keep],
            frames[frames['nodpos'] == 'B'].iloc[:n_keep],
        ]).sort_values('date_obs').reset_index(drop=True)

    dirname = f'{sanitize(obj)}_{setting}_{tpl_short}'
    raw_lines = [f'../../raw/{row["dp_id"]}.fits OBS_NODDING_OTHER\n'
                 for _, row in comb_frames.iterrows()]
    write_sofs(outdir / dirname, raw_lines, tw, flat, blaze)
    n_combined += 1

    # --- individual AB pairs ---
    paired = set()
    pairs = []
    for i in range(len(frames)):
        if i in paired:
            continue
        for j in range(i + 1, len(frames)):
            if j in paired:
                continue
            if frames.iloc[i]['nodpos'] != frames.iloc[j]['nodpos']:
                pairs.append((i, j))
                paired.add(i)
                paired.add(j)
                break
    n_skipped += len(frames) - len(paired)

    for pair_num, (i, j) in enumerate(pairs, 1):
        f1 = frames.iloc[i]
        f2 = frames.iloc[j]
        dirname = f'{sanitize(obj)}_{setting}_{tpl_short}_{pair_num}'
        raw_lines = [
            f'../../raw/{f1["dp_id"]}.fits OBS_NODDING_OTHER\n',
            f'../../raw/{f2["dp_id"]}.fits OBS_NODDING_OTHER\n',
        ]
        write_sofs(outdir / dirname, raw_lines, tw, flat, blaze)
        n_pairs += 1

print(f'{n_combined} combined template dirs')
print(f'{n_pairs} AB pair dirs')
print(f'{n_skipped} frames unpaired')
