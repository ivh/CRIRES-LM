# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyvo",
# ]
# ///
"""Update LMscience.sqlite with new public L/M-band science frames from ESO TAP."""

import sqlite3
from pathlib import Path
from pyvo import dal

DB_PATH = Path(__file__).parent / "LMscience.sqlite"

COLUMNS = [
    "dp_id", "object", "date_obs", "prog_id", "ob_id",
    "ins_wlen_id", "ins_wlen_cwlen", "ins_slit1_wid", "ins_filt1_name",
    "det_dit", "det_ndit", "dp_type", "dp_cat",
    "tpl_name", "tpl_start", "exposure", "tpl_id",
]

def update_db(until="2025-03-19"):
    db = sqlite3.connect(DB_PATH)
    last_date = db.execute("SELECT MAX(date_obs) FROM frames").fetchone()[0]
    print(f"DB has data up to {last_date}, querying from there to {until}")

    tap = dal.TAPService("https://archive.eso.org/tap_obs")
    col_str = ", ".join(COLUMNS)

    all_rows = []
    offset_date = last_date
    offset_dp = ""
    batch = 5000

    while True:
        result = tap.search(f"""
        SELECT TOP {batch} {col_str}, date_obs AS dobs
        FROM ist.crires
        WHERE dp_type = 'OBJECT'
          AND dp_cat = 'SCIENCE'
          AND (ins_wlen_id LIKE 'L%' OR ins_wlen_id LIKE 'M%')
          AND date_obs > '{offset_date}'
          AND date_obs < '{until}'
          AND dp_id > '{offset_dp}'
        ORDER BY date_obs, dp_id
        """).to_table()
        if len(result) == 0:
            break
        for row in result:
            all_rows.append(tuple(
                str(row[c]) if row[c] is not None else None for c in COLUMNS
            ))
        offset_date = str(result[-1]["dobs"])
        offset_dp = str(result[-1]["dp_id"])
        print(f"  queried {len(all_rows)} rows so far")

    if not all_rows:
        print("No new frames found.")
        db.close()
        return

    print(f"Inserting {len(all_rows)} new frames")
    db.executemany(
        f"INSERT OR IGNORE INTO frames ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
        all_rows,
    )
    db.commit()
    new_total = db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    print(f"Done. Total frames: {new_total}")
    db.close()

if __name__ == "__main__":
    update_db()
