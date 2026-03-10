# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "astroquery",
#     "astropy",
#     "pyvo",
# ]
# ///
"""Fetch all deep (NDIT>10) L/M-band CRIRES FLAT frames from 2021-01 to 2025-02."""

import sqlite3
import time
from pathlib import Path
from pyvo import dal
from astroquery.eso import Eso

DB_PATH = Path(__file__).parent / "deep_flats.sqlite"
DEST = Path(__file__).parent / "flats"

def build_db():
    tap = dal.TAPService("https://archive.eso.org/tap_obs")

    columns = [
        "dp_id", "date_obs", "ins_wlen_id", "ins_wlen_cwlen",
        "ins_filt1_name", "det_dit", "det_ndit", "dp_type",
        "tpl_name", "tpl_start", "ob_id", "prog_id"
    ]
    col_str = ", ".join(columns)

    all_rows = []
    offset_date = "2021-01-01"
    offset_dp = ""
    batch = 5000

    while True:
        result = tap.search(f"""
        SELECT TOP {batch} {col_str}, date_obs AS dobs
        FROM ist.crires
        WHERE dp_type = 'FLAT'
          AND dp_cat = 'CALIB'
          AND det_ndit > 10
          AND (ins_wlen_id LIKE 'L%' OR ins_wlen_id LIKE 'M%')
          AND date_obs >= '2021-01-01'
          AND date_obs < '2025-03-01'
          AND (date_obs > '{offset_date}' OR (date_obs = '{offset_date}' AND dp_id > '{offset_dp}'))
        ORDER BY date_obs, dp_id
        """).to_table()
        if len(result) == 0:
            break
        for row in result:
            all_rows.append(tuple(str(row[c]) if row[c] is not None else None for c in columns))
        offset_date = str(result[-1]["dobs"])
        offset_dp = str(result[-1]["dp_id"])
        print(f"  queried {len(all_rows)} rows so far")

    print(f"Total: {len(all_rows)} frames")

    db = sqlite3.connect(DB_PATH)
    db.execute("""
    CREATE TABLE IF NOT EXISTS flats (
        dp_id TEXT PRIMARY KEY,
        date_obs TEXT,
        ins_wlen_id TEXT,
        ins_wlen_cwlen REAL,
        ins_filt1_name TEXT,
        det_dit REAL,
        det_ndit INTEGER,
        dp_type TEXT,
        tpl_name TEXT,
        tpl_start TEXT,
        ob_id TEXT,
        prog_id TEXT,
        fetched INTEGER DEFAULT 0
    )
    """)
    db.executemany(
        f"INSERT OR IGNORE INTO flats ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
        all_rows,
    )
    db.commit()
    db.close()

def fetch_all():
    DEST.mkdir(exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    eso = Eso()

    pending = [
        row[0]
        for row in db.execute(
            "SELECT dp_id FROM flats WHERE fetched = 0 ORDER BY date_obs"
        ).fetchall()
    ]
    print(f"{len(pending)} frames to fetch")

    for i, dp_id in enumerate(pending, 1):
        fitsfile = DEST / f"{dp_id}.fits"

        if fitsfile.exists():
            db.execute("UPDATE flats SET fetched = 1 WHERE dp_id = ?", (dp_id,))
            db.commit()
            print(f"[{i}/{len(pending)}] {dp_id} already exists")
            continue

        print(f"[{i}/{len(pending)}] downloading {dp_id} ...", end=" ", flush=True)
        try:
            files = eso.retrieve_data([dp_id], destination=str(DEST))
            if files:
                db.execute("UPDATE flats SET fetched = 1 WHERE dp_id = ?", (dp_id,))
                db.commit()
                print("ok")
            else:
                print("no file returned")
        except Exception as e:
            print(f"FAILED: {e}")

        if i < len(pending):
            time.sleep(5)

    db.close()
    print("done")

if __name__ == "__main__":
    if not DB_PATH.exists():
        print("Building database...")
        build_db()
    fetch_all()
