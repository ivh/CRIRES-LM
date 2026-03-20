#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["fastapi", "uvicorn", "jinja2", "astropy", "plotly", "numpy", "markdown"]
# ///
"""CRIRES+ L/M-band data browser."""

import re
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import markdown
import numpy as np
import plotly
import plotly.graph_objects as go
from astropy.io import fits
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

BASE = Path(__file__).parent
REDUCED = BASE / "reduced"
REDUCED_NOFLAT = BASE / "reduced_noflat"
FLATS = BASE / "flats"


def _reduced_dir(variant=None):
    if variant == "noflat":
        return REDUCED_NOFLAT
    return REDUCED

# precomputed lookup: dirname -> tpl_start
_dir_to_tpl: dict[str, str] = {}
_tpl_to_dir: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app):
    init_db()
    build_dir_index()
    yield


app = FastAPI(title="CRIRES+ L/M Browser", lifespan=lifespan)
templates = Jinja2Templates(directory=str(BASE / "templates"))


@app.middleware("http")
async def add_base_url(request, call_next):
    """Make root_path available to templates as `base`."""
    base = request.scope.get("root_path", "")
    templates.env.globals["base"] = base
    return await call_next(request)


# --- Database ---

def get_db():
    conn = sqlite3.connect(BASE / "LMscience.sqlite")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = sqlite3.connect(BASE / "LMscience.sqlite")
    conn.executescript("""
        CREATE VIEW IF NOT EXISTS observations AS
        SELECT
            tpl_start,
            object,
            ins_wlen_id,
            ins_wlen_cwlen,
            ins_slit1_wid,
            prog_id,
            MIN(date_obs) as date_obs,
            COUNT(*) as n_frames,
            SUM(CASE WHEN nodpos='A' THEN 1 ELSE 0 END) as n_A,
            SUM(CASE WHEN nodpos='B' THEN 1 ELSE 0 END) as n_B,
            MAX(det_dit) as det_dit,
            MAX(det_ndit) as det_ndit,
            SUM(exposure) as total_exposure
        FROM frames
        WHERE nodpos IN ('A','B')
          AND tpl_name='Nodding along slit using jitter'
        GROUP BY tpl_start;
    """)
    conn.close()


def build_dir_index():
    conn = get_db()
    for row in conn.execute(
        "SELECT object, ins_wlen_id, tpl_start FROM observations"
    ):
        dirname = _make_dirname(row["object"], row["ins_wlen_id"], row["tpl_start"])
        _dir_to_tpl[dirname] = row["tpl_start"]
        _tpl_to_dir[row["tpl_start"]] = dirname
    conn.close()


# --- Helpers ---

def _sanitize(name):
    return re.sub(r'[^a-zA-Z0-9._+-]', '_', name)


def _make_dirname(obj, setting, tpl_start):
    tpl_short = tpl_start.replace('T', '_').replace(':', '')[:16]
    return f'{_sanitize(obj)}_{setting}_{tpl_short}'


def _dirpath(dirname, variant=None):
    return _reduced_dir(variant) / dirname


def reduction_status(dirname, variant=None):
    dp = _dirpath(dirname, variant)
    if not dp.exists():
        return {"exists": False, "extracted": False, "tellcorr": False, "wavecorr": False}
    return {
        "exists": True,
        "extracted": (dp / "cr2res_obs_nodding_extractedA.fits").exists(),
        "tellcorr": (dp / "cr2res_obs_nodding_extractedA_tellcorr.fits").exists(),
        "wavecorr": bool(list(dp.glob("wavecorr*.png"))),
    }


def read_spectra(dirname, nod="A", variant=None):
    """Read spectral data from tellcorr FITS only. Returns {order: [segments]}."""
    dp = _dirpath(dirname, variant)
    fitsfile = dp / f"cr2res_obs_nodding_extracted{nod}_tellcorr.fits"
    if not fitsfile.exists():
        return {}

    orders: dict[int, list] = {}
    with fits.open(fitsfile) as hdul:
        for chip in [1, 2, 3]:
            ext = f"CHIP{chip}.INT1"
            if ext not in hdul:
                continue
            data = hdul[ext].data
            cols = data.columns.names

            for col in cols:
                m = re.match(r'(\d+)_01_SPEC', col)
                if not m:
                    continue
                order_num = int(m.group(1))
                pfx = f"{m.group(1)}_01"

                wl = data[f"{pfx}_WL"] if f"{pfx}_WL" in cols else None
                spec = data[f"{pfx}_SPEC"] if f"{pfx}_SPEC" in cols else None
                if wl is None or spec is None:
                    continue

                err = data[f"{pfx}_ERR"] if f"{pfx}_ERR" in cols else None
                tellur = data[f"{pfx}_TELLUR"] if f"{pfx}_TELLUR" in cols else None
                cont = data[f"{pfx}_CONT"] if f"{pfx}_CONT" in cols else None

                mask = np.isfinite(wl) & (wl > 0) & np.isfinite(spec)
                if not np.any(mask):
                    continue

                seg = {
                    "chip": chip,
                    "wl": np.round(wl[mask], 4).tolist(),
                    "spec": np.round(spec[mask], 2).tolist(),
                }
                if err is not None:
                    seg["err"] = np.round(err[mask], 2).tolist()
                if tellur is not None and np.any(np.isfinite(tellur[mask])):
                    seg["tellur"] = np.round(tellur[mask], 4).tolist()
                if cont is not None and np.any(np.isfinite(cont[mask])):
                    seg["cont"] = np.round(cont[mask], 2).tolist()

                orders.setdefault(order_num, [])
                orders[order_num].append(seg)
    return orders


def make_spectrum_plot(dirname, variant=None):
    spec_a = read_spectra(dirname, "A", variant)
    spec_b = read_spectra(dirname, "B", variant)
    if not spec_a and not spec_b:
        return None

    has_tellur = any(
        any(s.get("tellur") for s in segs) for segs in spec_a.values()
    )

    fig = go.Figure()
    colors = plotly.colors.qualitative.D3
    all_orders = sorted(set(list(spec_a.keys()) + list(spec_b.keys())))

    # plot uncorrected spectra (SPEC * TELLUR) and model (CONT * TELLUR)
    # matching what the diagnostic PNGs show
    for i, onum in enumerate(all_orders):
        color = colors[i % len(colors)]

        if onum in spec_a:
            for seg in sorted(spec_a[onum], key=lambda s: s["chip"]):
                wl = seg["wl"]
                spec = seg["spec"]
                tellur = seg.get("tellur")
                cont = seg.get("cont")
                # reconstruct uncorrected = corrected * telluric
                if tellur:
                    y = [s * t for s, t in zip(spec, tellur)]
                else:
                    y = spec
                fig.add_trace(go.Scattergl(
                    x=wl, y=y,
                    mode="lines", line=dict(color=color, width=1),
                    name=f"A ord {onum:02d}",
                    legendgroup=f"A_{onum}",
                    showlegend=(seg["chip"] == 1),
                    hovertemplate=(
                        f"Ord {onum} Chip {seg['chip']}<br>"
                        "WL: %{x:.2f} nm<br>Flux: %{y:.1f}<extra>A</extra>"
                    ),
                ))
                # model = CONT * TELLUR
                if cont and tellur:
                    model = [c * t for c, t in zip(cont, tellur)]
                    fig.add_trace(go.Scattergl(
                        x=wl, y=model,
                        mode="lines", line=dict(color="black", width=1.5),
                        name="Model",
                        legendgroup="model",
                        showlegend=(i == 0 and seg["chip"] == 1),
                        hovertemplate=(
                            f"Ord {onum} Chip {seg['chip']}<br>"
                            "WL: %{x:.2f} nm<br>Model: %{y:.1f}<extra></extra>"
                        ),
                    ))

        if onum in spec_b:
            for seg in sorted(spec_b[onum], key=lambda s: s["chip"]):
                wl = seg["wl"]
                spec = seg["spec"]
                tellur = seg.get("tellur")
                cont = seg.get("cont")
                if tellur:
                    y = [s * t for s, t in zip(spec, tellur)]
                else:
                    y = spec
                fig.add_trace(go.Scattergl(
                    x=wl, y=y,
                    mode="lines", line=dict(color=color, width=1),
                    name=f"B ord {onum:02d}",
                    legendgroup=f"B_{onum}",
                    showlegend=(seg["chip"] == 1),
                    visible="legendonly",
                    hovertemplate=(
                        f"Ord {onum} Chip {seg['chip']}<br>"
                        "WL: %{x:.2f} nm<br>Flux: %{y:.1f}<extra>B</extra>"
                    ),
                ))
                if cont and tellur:
                    model = [c * t for c, t in zip(cont, tellur)]
                    fig.add_trace(go.Scattergl(
                        x=wl, y=model,
                        mode="lines", line=dict(color="black", width=1.5),
                        name="Model B",
                        legendgroup="model_b",
                        showlegend=(i == 0 and seg["chip"] == 1),
                        visible="legendonly",
                        hovertemplate=(
                            f"Ord {onum} Chip {seg['chip']}<br>"
                            "WL: %{x:.2f} nm<br>Model B: %{y:.1f}<extra></extra>"
                        ),
                    ))

    # auto-range y-axis using percentiles
    all_vals = []
    for segs in spec_a.values():
        for seg in segs:
            if seg.get("tellur"):
                all_vals.extend(s * t for s, t in zip(seg["spec"], seg["tellur"]))
            else:
                all_vals.extend(seg["spec"])
    if all_vals:
        arr = np.array(all_vals)
        lo, hi = np.nanpercentile(arr, [1, 99])
        margin = (hi - lo) * 0.1
        yrange = [lo - margin, hi + margin]
    else:
        yrange = None

    layout = dict(
        title="Spectrum with telluric model",
        xaxis_title="Wavelength (nm)",
        yaxis=dict(title="Flux (ADU)", range=yrange, autorange=False),
        height=550,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(font=dict(size=10)),
        hovermode="closest",
        margin=dict(t=40, b=40),
    )
    fig.update_layout(**layout)
    fig.layout.template = None
    return fig


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    q: str = Query(None),
    object: str = Query(None),
    setting: str = Query(None),
    prog_id: str = Query(None),
    variant: str = Query(None),
):
    conn = get_db()

    objects = [r[0] for r in conn.execute(
        "SELECT DISTINCT object FROM observations ORDER BY object"
    )]
    settings = [r[0] for r in conn.execute(
        "SELECT DISTINCT ins_wlen_id FROM observations ORDER BY ins_wlen_id"
    )]
    prog_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT prog_id FROM observations ORDER BY prog_id"
    )]

    query = "SELECT * FROM observations WHERE 1=1"
    params: list = []
    if q:
        query += " AND (object LIKE ? OR prog_id LIKE ? OR tpl_start LIKE ?)"
        like = f"%{q}%"
        params.extend([like, like, like])
    if object:
        query += " AND object = ?"
        params.append(object)
    if setting:
        query += " AND ins_wlen_id = ?"
        params.append(setting)
    if prog_id:
        query += " AND prog_id = ?"
        params.append(prog_id)
    query += " ORDER BY date_obs DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    observations = []
    for row in rows:
        d = dict(row)
        d["dirname"] = _tpl_to_dir.get(d["tpl_start"], "")
        d.update(reduction_status(d["dirname"], variant))
        if not d["tellcorr"]:
            continue
        observations.append(d)

    variant_qs = f"?variant={variant}" if variant else ""

    ctx = {
        "request": request,
        "observations": observations,
        "objects": objects,
        "settings": settings,
        "prog_ids": prog_ids,
        "sel_q": q,
        "sel_object": object,
        "sel_setting": setting,
        "sel_prog_id": prog_id,
        "n_total": len(observations),
        "variant": variant,
        "variant_qs": variant_qs,
    }

    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("obs_table.html", ctx)
    return templates.TemplateResponse("index.html", ctx)


def _resolve_dirname(dirname):
    """Resolve dirname to (tpl_start, base_dirname, pair_num).

    For combined dirs (no suffix): pair_num=None.
    For pair dirs (_N suffix): strip suffix to find tpl_start.
    """
    if dirname in _dir_to_tpl:
        return _dir_to_tpl[dirname], dirname, None
    m = re.match(r'^(.+)_(\d+)$', dirname)
    if m:
        base, pair_num = m.group(1), int(m.group(2))
        if base in _dir_to_tpl:
            return _dir_to_tpl[base], base, pair_num
    return None, None, None


def _pair_frames(frames):
    """Greedy AB pairing, same logic as make_reduction_sofs.py.

    Returns list of (pair_num, frame_a, frame_b) and list of unpaired frames.
    """
    paired = set()
    pairs = []
    for i in range(len(frames)):
        if i in paired:
            continue
        for j in range(i + 1, len(frames)):
            if j in paired:
                continue
            if frames[i]["nodpos"] != frames[j]["nodpos"]:
                pairs.append((i, j))
                paired.add(i)
                paired.add(j)
                break
    unpaired = [frames[i] for i in range(len(frames)) if i not in paired]
    result = []
    for pair_num, (i, j) in enumerate(pairs, 1):
        result.append((pair_num, frames[i], frames[j]))
    return result, unpaired


@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    readme = (BASE / "README.md").read_text()
    html = markdown.markdown(readme, extensions=["fenced_code"])
    return templates.TemplateResponse("about.html", {
        "request": request,
        "content": html,
    })


@app.get("/flats", response_class=HTMLResponse)
def flat_index(request: Request, setting: str = Query(None)):
    flat_dirs = sorted(
        d.name for d in FLATS.iterdir()
        if d.is_dir() and d.name not in ("raw", "sof")
    )
    if setting:
        flat_dirs = [d for d in flat_dirs if d.startswith(setting)]
    settings = sorted({d.rsplit("_", 1)[0] for d in flat_dirs})
    return templates.TemplateResponse("flats_index.html", {
        "request": request,
        "flat_dirs": flat_dirs,
        "settings": settings,
        "sel_setting": setting,
    })


@app.get("/flat/{dirname}", response_class=HTMLResponse)
def flat_detail(request: Request, dirname: str):
    flatdir = FLATS / dirname
    if not flatdir.is_dir():
        return HTMLResponse("Flat not found", status_code=404)

    parts = dirname.rsplit("_", 1)
    setting = parts[0] if len(parts) == 2 else dirname
    date = parts[1] if len(parts) == 2 else ""

    # parse SOF for raw frame list, look up DIT/NDIT from deep_flats.sqlite
    sof_file = flatdir / "flats.sof"
    raw_frames = []
    if sof_file.exists():
        dp_ids = []
        for line in sof_file.read_text().splitlines():
            if not line.strip():
                continue
            filename, tag = line.strip().rsplit(None, 1)
            name = Path(filename).name
            dp_id = name.removesuffix(".fits")
            raw_frames.append({"filename": name, "tag": tag, "dp_id": dp_id})
            if tag == "FLAT":
                dp_ids.append(dp_id)
        if dp_ids:
            conn_flats = sqlite3.connect(BASE / "deep_flats.sqlite")
            conn_flats.row_factory = sqlite3.Row
            placeholders = ",".join("?" * len(dp_ids))
            rows = conn_flats.execute(
                f"SELECT dp_id, det_dit, det_ndit FROM flats WHERE dp_id IN ({placeholders})",
                dp_ids,
            ).fetchall()
            conn_flats.close()
            dit_ndit = {r["dp_id"]: (r["det_dit"], r["det_ndit"]) for r in rows}
            for f in raw_frames:
                if f["dp_id"] in dit_ndit:
                    f["det_dit"], f["det_ndit"] = dit_ndit[f["dp_id"]]

    # read basic header info from master flat if present
    header_info = {}
    master = flatdir / "cr2res_cal_flat_Open_master_flat.fits"
    if master.exists():
        with fits.open(master) as hdul:
            h = hdul[0].header
            for key, label in [
                ("ESO INS WLEN ID", "Setting"),
                ("ESO DET DIT", "DIT"),
                ("ESO DET NDIT", "NDIT"),
                ("ESO INS SLIT1 WID", "Slit width"),
                ("ESO INS FILT1 NAME", "Filter"),
            ]:
                if key in h:
                    header_info[label] = h[key]

    images = sorted(p.name for p in flatdir.glob("*.png"))

    products = sorted(
        p.name for p in flatdir.iterdir()
        if p.suffix == ".fits"
    )

    return templates.TemplateResponse("flat_detail.html", {
        "request": request,
        "dirname": dirname,
        "setting": setting,
        "date": date,
        "header_info": header_info,
        "raw_frames": raw_frames,
        "images": images,
        "products": products,
    })


@app.get("/flat-files/{dirname}/{filename}")
def serve_flat_file(dirname: str, filename: str):
    filepath = FLATS / dirname / filename
    if not filepath.exists() or not filepath.is_file():
        return HTMLResponse("Not found", status_code=404)
    if filename.endswith(".png"):
        return FileResponse(filepath, media_type="image/png")
    if filename.endswith(".fits"):
        return FileResponse(filepath, media_type="application/octet-stream", filename=filename)
    return HTMLResponse("Unsupported file type", status_code=400)


@app.get("/obs/{dirname}", response_class=HTMLResponse)
def observation(request: Request, dirname: str, variant: str = Query(None)):
    tpl_start, base_dirname, pair_num = _resolve_dirname(dirname)
    if not tpl_start:
        return HTMLResponse("Observation not found", status_code=404)

    conn = get_db()
    row = conn.execute(
        "SELECT * FROM observations WHERE tpl_start = ?", [tpl_start]
    ).fetchone()
    if not row:
        conn.close()
        return HTMLResponse("Observation not found", status_code=404)

    obs = dict(row)
    obs["dirname"] = dirname
    obs["base_dirname"] = base_dirname
    obs["pair_num"] = pair_num
    st = reduction_status(dirname, variant)
    obs.update(st)

    frames = conn.execute(
        "SELECT dp_id, nodpos, date_obs, det_dit, det_ndit, exposure "
        "FROM frames WHERE tpl_start = ? ORDER BY date_obs",
        [tpl_start],
    ).fetchall()
    frame_list = [dict(f) for f in frames]

    pairs, unpaired = _pair_frames(frame_list)

    # for pair pages, only show the two frames of that pair
    if pair_num is not None:
        obs["frames"] = []
        for pn, fa, fb in pairs:
            if pn == pair_num:
                obs["frames"] = [fa, fb]
                break
        obs["pairs"] = []
        obs["unpaired"] = []
    else:
        obs["frames"] = frame_list
        obs["pairs"] = pairs
        obs["unpaired"] = unpaired

    # prev/next within same setting, only those with tellcorr data
    neighbors = conn.execute(
        "SELECT tpl_start FROM observations "
        "WHERE ins_wlen_id = ? ORDER BY date_obs",
        [obs["ins_wlen_id"]],
    ).fetchall()
    tpl_list = [
        r[0] for r in neighbors
        if reduction_status(_tpl_to_dir.get(r[0], ""), variant).get("tellcorr")
    ]
    idx = tpl_list.index(tpl_start) if tpl_start in tpl_list else -1
    prev_dir = _tpl_to_dir.get(tpl_list[idx - 1]) if idx > 0 else None
    next_dir = _tpl_to_dir.get(tpl_list[idx + 1]) if idx < len(tpl_list) - 1 else None
    conn.close()

    dp = _dirpath(dirname, variant)
    images = sorted(p.name for p in dp.glob("*.png")) if dp.exists() else []
    downloads = sorted(
        p.name for p in dp.iterdir()
        if p.name.endswith("_tellcorr.fits")
    ) if dp.exists() else []

    # extract flat directory from calib.sof (or nodd.sof for older layouts)
    flat_dirname = None
    for sof_name in ("calib.sof", "nodd.sof"):
        sof = dp / sof_name if dp.exists() else None
        if sof and sof.exists():
            for line in sof.read_text().splitlines():
                if "CAL_FLAT_MASTER" in line:
                    flat_dirname = Path(line.split()[0]).parent.name
                    break
            if flat_dirname:
                break

    variant_qs = f"?variant={variant}" if variant else ""
    other_variant = None if variant else "noflat"
    other_variant_qs = f"?variant={other_variant}" if other_variant else ""
    other_variant_label = "with flat" if variant else "without flat"

    return templates.TemplateResponse("observation.html", {
        "request": request,
        "obs": obs,
        "images": images,
        "downloads": downloads,
        "prev_dir": prev_dir,
        "next_dir": next_dir,
        "flat_dirname": flat_dirname,
        "variant": variant,
        "variant_qs": variant_qs,
        "other_variant_qs": other_variant_qs,
        "other_variant_label": other_variant_label,
    })


@app.get("/api/spectrum/{dirname}")
def api_spectrum(dirname: str, variant: str = Query(None)):
    tpl_start, _, _ = _resolve_dirname(dirname)
    if not tpl_start:
        return HTMLResponse("Not found", status_code=404)
    st = reduction_status(dirname, variant)
    if not st["tellcorr"]:
        return {"data": [], "layout": {}}
    fig = make_spectrum_plot(dirname, variant)
    if not fig:
        return {"data": [], "layout": {}}
    return fig.to_plotly_json()


@app.get("/sw.js")
def unregister_sw():
    """Serve a self-unregistering service worker to clear stale SW from other apps."""
    from fastapi.responses import Response
    return Response(
        "self.addEventListener('install', () => self.skipWaiting());\n"
        "self.addEventListener('activate', () => self.registration.unregister());\n",
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/files/{dirname}/{filename}")
def serve_file(dirname: str, filename: str, variant: str = Query(None)):
    tpl_start, _, _ = _resolve_dirname(dirname)
    if not tpl_start:
        return HTMLResponse("Not found", status_code=404)
    filepath = _reduced_dir(variant) / dirname / filename
    if not filepath.exists() or not filepath.is_file():
        return HTMLResponse("Not found", status_code=404)

    if filename.endswith(".png"):
        return FileResponse(filepath, media_type="image/png")
    if filename.endswith(".fits"):
        return FileResponse(
            filepath,
            media_type="application/octet-stream",
            filename=filename,
        )
    return HTMLResponse("Unsupported file type", status_code=400)


if __name__ == "__main__":
    import argparse
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--root-path", default="")
    args = p.parse_args()
    uvicorn.run("webapp:app", host="0.0.0.0", port=args.port,
                reload=args.reload, root_path=args.root_path)
