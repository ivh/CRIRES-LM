#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas"]
# ///
"""Generate directory structure and SOF files for reducing all science AB pairs."""

import re
import sqlite3
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent

conn = sqlite3.connect(BASE / 'LMscience.sqlite')
df = pd.read_sql('''
    SELECT dp_id, object, ins_wlen_id, tpl_start, nodpos, date_obs, tpl_name
    FROM frames
    WHERE nodpos IN ('A', 'B')
      AND tpl_name = 'Nodding along slit using jitter'
    ORDER BY tpl_start, date_obs
''', conn)
conn.close()

print(f'{len(df)} frames with nod positions')


def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9._+-]', '_', name)


outdir = BASE / 'reduced'
outdir.mkdir(exist_ok=True)

n_pairs = 0
n_skipped = 0

for tpl_start, group in df.groupby('tpl_start'):
    setting = group.iloc[0]['ins_wlen_id']
    obj = group.iloc[0]['object']

    tw = f'./{setting}_tw.fits'
    flat = f'../../flats/{setting}/cr2res_cal_flat_Open_master_flat.fits'

    frames = group.sort_values('date_obs').reset_index(drop=True)

    # greedy pairing: walk through frames, match each A with next B or vice versa
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

        tpl_short = tpl_start.replace('T', '_').replace(':', '')[:16]
        dirname = f'{sanitize(obj)}_{setting}_{tpl_short}_{pair_num}'
        pairdir = outdir / dirname
        pairdir.mkdir(exist_ok=True)

        sof = pairdir / 'nodd.sof'
        with open(sof, 'w') as f:
            f.write(f'../../raw/{f1["dp_id"]}.fits OBS_NODDING_OTHER\n')
            f.write(f'../../raw/{f2["dp_id"]}.fits OBS_NODDING_OTHER\n')
            f.write(f'{tw} UTIL_WAVE_TW\n')
            f.write(f'{flat} CAL_FLAT\n')

        n_pairs += 1

print(f'{n_pairs} AB pairs created')
print(f'{n_skipped} frames skipped (odd or same nod position)')

# summary by setting
pair_dirs = sorted(outdir.iterdir())
by_setting = {}
for d in pair_dirs:
    if not d.is_dir():
        continue
    parts = d.name.split('_')
    # setting is the part matching L\d{4} or M\d{4}
    for p in parts:
        if re.match(r'^[LM]\d{4}$', p):
            by_setting.setdefault(p, 0)
            by_setting[p] += 1
            break

print('\nPairs per setting:')
for s in sorted(by_setting):
    print(f'  {s}: {by_setting[s]}')
