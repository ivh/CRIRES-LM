# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "astroquery",
#     "astropy",
# ]
# ///
"""Fetch all L/M-band CRIRES science frames listed in LMscience.sqlite,
fill the nodpos column from FITS headers after download."""

import sqlite3
import time
import sys
from pathlib import Path
from astroquery.eso import Eso
from astropy.io import fits

DB_PATH = Path(__file__).parent / "LMscience.sqlite"
DEST = Path(__file__).parent / "raw"

def get_pending(db):
    cur = db.execute(
        "SELECT dp_id FROM frames WHERE nodpos IS NULL ORDER BY date_obs"
    )
    return [row[0] for row in cur.fetchall()]

def fill_nodpos(db, dp_id, filepath):
    h = fits.getheader(filepath)
    nodpos = h.get("ESO SEQ NODPOS")
    db.execute("UPDATE frames SET nodpos = ? WHERE dp_id = ?", (nodpos, dp_id))
    db.commit()
    return nodpos

def main():
    DEST.mkdir(exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    eso = Eso()

    pending = get_pending(db)
    print(f"{len(pending)} frames to fetch")

    for i, dp_id in enumerate(pending, 1):
        fitsfile = DEST / f"{dp_id}.fits"

        if fitsfile.exists():
            nodpos = fill_nodpos(db, dp_id, fitsfile)
            print(f"[{i}/{len(pending)}] {dp_id} already exists, nodpos={nodpos}")
            continue

        print(f"[{i}/{len(pending)}] downloading {dp_id} ...", end=" ", flush=True)
        try:
            files = eso.retrieve_data([dp_id], destination=str(DEST))
            if files:
                nodpos = fill_nodpos(db, dp_id, files[0])
                print(f"ok, nodpos={nodpos}")
            else:
                print("no file returned")
        except Exception as e:
            print(f"FAILED: {e}")

        if i < len(pending):
            time.sleep(5)

    db.close()
    print("done")

if __name__ == "__main__":
    main()
