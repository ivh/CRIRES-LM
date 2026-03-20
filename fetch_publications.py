# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""Fetch publications from ESO telbib API for all CRIRES L/M program IDs."""

import sqlite3
import httpx
import time
import json
import re
import xml.etree.ElementTree as ET

def get_program_ids():
    con = sqlite3.connect("LMscience.sqlite")
    rows = con.execute("SELECT DISTINCT prog_id FROM frames ORDER BY prog_id").fetchall()
    con.close()
    return [r[0] for r in rows]

def parse_telbib_xml(xml_text):
    """Parse telbib API v2 XML response into list of publication dicts."""
    root = ET.fromstring(xml_text)
    pubs = []
    for item in root.iter("item"):
        pub = {}
        for field in ["bibcode", "title", "journal", "year", "volume", "pages", "doi"]:
            el = item.find(field)
            if el is not None and el.text:
                val = el.text.strip()
                if field == "title":
                    val = re.sub(r"<[^>]+>", "", val)
                pub[field] = val
        # authors: nested <author> elements
        authors_el = item.find("authors")
        if authors_el is not None:
            names = [a.text.strip() for a in authors_el.findall("author") if a.text]
            if names:
                if len(names) > 3:
                    pub["authors"] = f"{names[0]} et al."
                else:
                    pub["authors"] = "; ".join(names)
                pub["author_count"] = len(names)
        if pub.get("bibcode"):
            pubs.append(pub)
    return pubs

def main():
    prog_ids = get_program_ids()
    # unique base program IDs
    seen = {}
    for pid in prog_ids:
        if pid.startswith("60."):
            seen[pid] = [pid]
        else:
            base = pid.rsplit(".", 1)[0]
            seen.setdefault(base, []).append(pid)

    print(f"Querying telbib for {len(seen)} programs...")

    all_results = {}
    with httpx.Client() as client:
        for i, (base, sub_ids) in enumerate(sorted(seen.items())):
            url = f"https://telbib.eso.org/api_v2.php?programid={base}"
            resp = client.get(url, follow_redirects=True, timeout=30)
            pubs = parse_telbib_xml(resp.text)
            if pubs:
                all_results[base] = {"sub_ids": sub_ids, "publications": pubs}
                print(f"  {base}: {len(pubs)} publications")
            time.sleep(0.3)

    # write markdown
    with open("LMpublications.md", "w") as f:
        f.write("# Publications using CRIRES+ L/M-band data\n\n")
        f.write(f"Source: [ESO Telescope Bibliography (telbib)](https://telbib.eso.org/)\n\n")

        n_total = sum(len(v["publications"]) for v in all_results.values())
        n_prog = len(all_results)
        f.write(f"{n_total} publications from {n_prog} programmes (out of {len(seen)} total).\n\n")

        for base, info in sorted(all_results.items()):
            sub_str = ", ".join(info["sub_ids"])
            f.write(f"## {sub_str}\n\n")
            for pub in sorted(info["publications"], key=lambda p: p.get("year", ""), reverse=True):
                authors = pub.get("authors", "")
                title = pub.get("title", "")
                journal = pub.get("journal", "")
                year = pub.get("year", "")
                vol = pub.get("volume", "")
                pages = pub.get("pages", "")
                bibcode = pub.get("bibcode", "")
                doi = pub.get("doi", "")

                ref_parts = [journal]
                if vol:
                    ref_parts.append(vol)
                if pages:
                    ref_parts.append(pages.rstrip("-"))
                ref_str = ", ".join(p for p in ref_parts if p)

                ads_url = f"https://ui.adsabs.harvard.edu/abs/{bibcode}"
                f.write(f"- {authors} ({year}), \"{title}\", {ref_str}. [{bibcode}]({ads_url})")
                if doi:
                    f.write(f" [doi:{doi}](https://doi.org/{doi})")
                f.write("\n")
            f.write("\n")

    print(f"\nWrote LMpublications.md: {n_total} publications from {n_prog} programmes")

if __name__ == "__main__":
    main()
