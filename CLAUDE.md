# CRIRES+ L/M-band reduction project

## Overview
Reduction of all public CRIRES+ L-band (2.8-4.2 um) and M-band (4.2-5.5 um) science data from 2021-01 to 2025-02. Includes flat-fielding, nodding extraction, telluric correction via vipere, and slit tilt characterization.

## CRIRES+ instrument basics
- 3 detector chips: CHIP1, CHIP2, CHIP3 — each 2048x2048 pixels
- Raw file structure: primary HDU (header only) + CHIP{1,2,3}.INT1 extensions
- Echelle spectrograph: multiple spectral orders per chip, 5-7 orders depending on wavelength setting
- Spatial pixel scale: ~0.059"/pixel (measured from nod throw fitting)
- Slit lengths: 10" standard
- Each raw frame is ~49 MB

## Local data files
- `LMscience.sqlite` — metadata for all L/M-band science frames (11,135 rows). Columns: dp_id, object, date_obs, prog_id, ob_id, ins_wlen_id, ins_wlen_cwlen, ins_slit1_wid, ins_filt1_name, det_dit, det_ndit, dp_type, dp_cat, tpl_name, tpl_start, exposure, tpl_id, nodpos
- `deep_flats.sqlite` — metadata for deep L/M flats NDIT>10 same time range (914 rows)
- `fetch_LM.py` — downloads science frames from LMscience.sqlite, fills nodpos from headers
- `fetch_deep_flats.py` — downloads deep flat frames from deep_flats.sqlite
- `reduce_flats.sh` — runs `esorex cr2res_cal_flat` on all SOFs in `sof/` via GNU parallel
- `sof/` — SOF files for flat reduction, named `{setting}_{date}_flats.sof`
- `flats/` — fetched raw flats in `raw/`, reduced output in `{setting}/` subdirs
- `raw/` — all fetched science frames (on server)
- `{setting}_tw.fits` — tracing/wavelength tables per setting, updated with vipere wavelengths and slit tilt
- `make_reduction_sofs.py` — generates `reduced/` directory tree with SOF files for all 5239 AB pairs
- `make_combined_sofs.py` — generates combined SOF files (all frames per template start in one SOF), balances A/B counts, skips single-nod sequences
- `reduced/{object}_{setting}_{tpl_start}_{pair}/` — one dir per AB pair, each with `nodd.sof`
- `reduced/{object}_{setting}_{tpl_start}/` — combined dir (all frames from one template), no `_N` suffix

## esorex cr2res usage
- `esorex cr2res_cal_flat sof` — reduce flats (run from dir containing the data)
- `esorex cr2res_obs_nodding sof` — reduce nodding observations
- SOF paths are relative to the directory esorex is called from, NOT relative to the SOF file
- Flat output products: `cr2res_cal_flat_Open_master_flat.fits`, `_bpm.fits`, `_blaze.fits`, `_slit_func.fits`, `_slit_model.fits`, `_tw.fits`
- Multiple reductions in same output dir overwrite each other — use separate dirs or run from inside subdirs
- SOF format: `filename TAG` per line. Tags: `FLAT`, `OBS_NODDING_OTHER`, `UTIL_WAVE_TW`, `CAL_FLAT`, `UTIL_TRACE`
- Batch run: `ls reduced/*/nodd.sof | parallel --bar -j 8 'cd {//} && esorex cr2res_obs_nodding nodd.sof 2>&1 > esorex.log'`

## CRIRES calibration OBs
- L-band flat/dark calibration OB: `60.A-9051(A)`, ob_id `3346506`
- This is a long-lived reusable OB (data from 2022 onwards)
- Calibration sequence per wavelength setting: LAMP,METROLOGY (NDIT=1-3) -> FLAT -> DARK
- The `ob_id` column links all frames in an observing block
- Deep flats use NDIT=50 (L-band) or NDIT=20 (M-band), but associated DARKs only have NDIT=3 — DIT/NDIT mismatch

## CRIRES nod position header keywords
- `ESO SEQ NODPOS`: A or B (NOT in TAP, only in FITS headers)
- `ESO SEQ NODTHROW`: nod throw in arcsec (typically 5.5 or 6.0)
- Nod pattern is usually ABBA but can vary — don't assume from tpl_expno
- The commanded nod throw is inaccurate — actual throw is 2-3% larger than commanded for 6.0" settings

## CRIRES science templates
- `CRIRES_spec_obs_AutoNodOnSlit` — standard ABBA nodding, ~99% of L/M science
- `CRIRES_spec_obs_GenericOffset` — spatial mapping
- `CRIRES_spec_obs_SpectroAstrometry` — spectroastrometry
- CRIRES science uses `dp_type = 'OBJECT'` for everything (no `STD` type); telluric standards must be identified by target name or programme

## Telluric standard stars (one A+B pair per setting, in tellurics/)
All hot B/O-type stars with clean continua in L/M band:
- L3244-L3302, L3412-L3426: alf Eri (B6), 0.20" slit, nod throw 6"
- L3340: HD 153426 (O9II), 0.40" slit, nod throw 6" (from prog 113.26PE.001)
- L3377, M4318: tet Aql (B9.5), 0.40" slit, nod throw 5.5"
- M4187, M4266: HR 7950 (B), 0.20" slit, nod throw 6"
- M4211: omi Hya (B9III), 0.20" slit, nod throw 6"
- M4368: bet Ori (B8), 0.20" slit, nod throw 6"
- M4416-M4504: alf Eri (B6), 0.20" slit, nod throw 6"
- M4519: V S CrA (T Tauri — only available target), 0.20" slit; poor but usable vipere fits

## Wavelength coverage per setting
- L3244: 2869-3982 nm, L3262: 2886-4003 nm, L3302: 2923-4050 nm
- L3340: 2960-4082 nm, L3377: 2843-4136 nm, L3412: 2873-4177 nm, L3426: 2886-4193 nm
- M4187: 3584-5504 nm, M4211: 3381-5535 nm, M4266: 3427-5604 nm
- M4318: 3470-5154 nm, M4368: 3513-5211 nm, M4416: 3552-5264 nm
- M4461: 3614-5314 nm, M4504: 3870-5364 nm, M4519: ~3435-5370 nm

## Number of orders per setting (for vipere -oset)
- 7 orders/chip: L3377, L3412, L3426, M4266, M4519
- 6 orders/chip: L3244, L3262, L3302, L3340, M4187, M4318, M4368, M4416
- 5 orders/chip: M4211 (also has missing CHIP1 order 8 — use -oset 2:16), M4461, M4504

## tellurics/ directory structure
- `tellurics/{setting}/` — one subdir per wavelength setting (16 settings: L3244-L3426, M4187-M4519)
- Each contains: raw A+B FITS, `nodd.sof`, esorex output, vipere output
- `nodd.sof` format: raw fits as `./CRIRE...fits OBS_NODDING_OTHER`, calibs as `../../{file} TAG`
- Run esorex from inside the subdir: `cd {setting} && esorex cr2res_obs_nodding nodd.sof`
- vipere output: `telluricA.par.dat`, `telluricA.rvo.dat`, `telluricB.par.dat`, `telluricB.rvo.dat`

### Scripts in tellurics/
- `measure_nod_throw.py` — measures actual nod throw by Gaussian fitting to combinedA spatial profiles
- `fit_tilt.py` — computes tilt from vipere A-B wavelength differences, groups by echelle order
- `fit_tilt_linear.py` — fits simple linear tilt(wavelength) per band (L and M separately)
- `set_slit_tilt.py` — writes SlitPolyB values into all _tw.fits files from linear tilt fit
- `set_wavelength.py` — updates Wavelength polynomials in _tw.fits from vipere fits
- `make_tw_origin.py` — generates `tw_origin.md` provenance table
- `tw_origin.md` — tracks which trace has vipere vs pipeline wavelength and tilt source
- `measure_tilt.py`, `plot_tilt_vs_wl.py`, `plot_all_wavecal.py` — earlier exploration scripts
- `xcorr_tilt.py` — cross-correlation tilt measurement (tried, not used in final calibration)

## vipere — telluric fitting tool
- Installed from `~/vipere.git`, run via `uvx --from ~/vipere.git vipere`
- Fits telluric absorption + wavelength solution + continuum to extracted CRIRES spectra
- Input: `cr2res_obs_nodding_extracted{A,B}.fits` — one file at a time
- Key options: `-oset 1:N` (order range), `-o TAG` (output prefix), `-plot 0` (no GUI plots), `-vcut N` (edge trim in km/s, default 100)
- Output files: `TAG.par.dat` (per-order fit parameters), `TAG.rvo.dat` (per-spectrum RV summary), `res/` (residuals)
- `plt.ion()` at module level causes GUI windows even with `-plot 0` — set `MPLBACKEND=Agg` in subprocess env to suppress
- Default config (`config_vipere.yaml`): `deg_norm=2`, `deg_wave=2` (overrides argparse defaults of 3)

### vipere order numbering
- `-oset` indices combine chip and spectral order: `order_idx, detector = divmod(order-1, 3); detector += 1`
- Order 1 = chip1 of HIGHEST spectral order, order 2 = chip2 of same, order 3 = chip3, order 4 = chip1 of next lower, etc.
- Inside `Spectrum()`: `n_orders = max(order_numbers_in_FITS)` (NOT count of orders), `order_drs = n_orders - order_idx`
- IMPORTANT: vipere uses the max DRS order number from FITS columns, not the count. For M4504 with orders 03-07, `n_orders=7` not 5. Using the count shifts the mapping by the difference.

### vipere wavelength model
- Wavelength = `poly(pixel - xcen, [wave0, wave1, wave2])` where xcen = mean(good_pixels) + 18
- `wave0, wave1, wave2` in par.dat are ascending polynomial coefficients (constant, linear, quadratic)
- Wavelengths are in Angstrom (input WL columns in nm are converted x10 inside vipere)
- To reconstruct: `wl = np.poly1d([wave2, wave1, wave0])(pixel_ok - xcen)`
- To convert to _tw.fits format (nm, pixel-centered): expand around pixel 0 and divide by 10

### vipere par.dat columns
`BJD n order chunk rv e_rv norm0-2 wave0-2 ip0 atm0-5 bkg0 prms`
- Each row = one order segment; `order` = vipere order index; last row may have `order=0` (summary)
- `prms` = percentage RMS of residuals (fit quality, lower is better); negative = failed fit
- `norm0-2` = continuum polynomial coefficients, evaluated as `polyval(pixel - xcen, [norm0, norm1, norm2])`
- `wave0-2` = fitted wavelength polynomial coefficients (Angstrom, centered on xcen)
- `atm0-5` = telluric molecule column density scalings (H2O, O2, CO, CO2, CH4, N2O)
- `xcen = mean(pixel_ok) + 18` — center for polynomial evaluation

### vipere forward model
- `S_model(pixel) = norm(pixel-xcen) * IP_conv(S_star(wl) * telluric(wl) + bkg)`
- Without stellar template: `S_star = 1`, so `S_model = norm * IP_conv(telluric)`
- Telluric transmission: `product over molecules of (stdAtmos_mol ** atm_coeff)`
- Residuals in `res/NNN_OOO.dat`: two columns `pixel residual`, where `residual = observed - model`
- Model can be reconstructed as `model = observed[pixel_ok] - residual`

### vipere edge trimming (`-vcut`)
- Default `-vcut 100` trims pixels within 100 km/s of atmospheric model wavelength edges
- This removes ~60-130 pixels per detector edge depending on order
- `-vcut 0` recovers full detector coverage; safe for L/M band where atm models cover the full range

### vipere L/M band support
- Added L and M bands to `bands_all` and `wave_band` arrays in `__init__.py` (~line 1202)
- Atmospheric model files: `vipere/lib/atmos/stdAtmos_{L,M}.fits` — copied from `/Users/tom/viper-web/molecfit/`
- Without these files, vipere cannot fit tellurics at wavelengths > 2.5 um (BIC scores identical)
- Vipere cannot fit the CO2 4.26 um fundamental band — atmospheric model has zero transmission there

### uvx caching
- `uvx` caches aggressively — after editing vipere source, must run `uv cache prune` to force rebuild

## Nod throw calibration

### Method
- Fit double Gaussians to |combinedA| spatial profiles at detector center (x=1024, 20-column median)
- `measure_nod_throw.py` outputs `nod_throw_measurements.json` (264 raw, 223 pass quality filter)
- Quality filter: dy_err < 0.2, per-setting sigma clipping (5*MAD)
- PIXEL_SCALE = 0.059 arcsec/pixel

### Results
- Commanded 6.0" settings: dy ~104-105 pixels (actual ~6.15")
- Commanded 5.5" settings: dy ~96 pix (M4318) or ~99 pix (L3377)
- Actual throw is 2-3% larger than commanded for most settings

## Slit tilt calibration

### Method
1. Run vipere separately on extractedA and extractedB for all 16 telluric settings
2. Compare fitted wavelength scales — A-B wavelength difference at each pixel reflects slit tilt
3. Convert to pixel shift, divide by measured nod throw (dy per setting)
4. Fit linear tilt(wavelength) per band with iterative 3-sigma clipping

### Linear tilt fits (`fit_tilt_linear.py`, `tilt_linear_fit.json`)
- **L band**: tilt = -6.97e-5 * wl_nm + 0.203, rms = 0.0049 (1096 pts, 24 rejected)
- **M band**: tilt = -4.68e-5 * wl_nm + 0.169, rms = 0.0058 (849 pts, 71 rejected)
- Within-chip tilt variation (~0.005 rms) is ignored — constant per trace
- The 4200-4500 nm CO2 gap is interpolated by the linear fit

### SlitPolyB in _tw.fits (`set_slit_tilt.py`)
- `SlitPolyB = [tilt, 0, 0]` for each trace, evaluated from linear fit at trace center wavelength
- Values range from +0.01 (bluest L, ~2850 nm) to -0.09 (reddest M, ~5500 nm)

## Wavelength calibration update

### Method (`set_wavelength.py`)
- Convert vipere polynomial (Angstrom, centered on xcen) to _tw.fits format (nm, centered on pixel 0)
- Quality filter: prms > 0 AND prms < 10, shift from pipeline < 10 km/s (or < 40 km/s if prms < 5)
- For rejected orders: restore wavelength from extracted spectrum WL column (pipeline calibration)

### Results
- 225/311 traces updated with vipere wavelengths (72%)
- 86 traces keep pipeline wavelength (mostly M-band long-wavelength orders with poor fits, CO2 gap orders)
- Typical vipere correction: 1-4 km/s from pipeline
- Provenance tracked in `tellurics/tw_origin.md`

### _tw.fits tracing table structure
- Extensions: CHIP1.INT1, CHIP2.INT1, CHIP3.INT1 — one row per spectral order
- Columns: `All`, `Upper`, `Lower` (trace polynomials: Y as f(X)), `Order`, `TraceNb`, `Wavelength`, `Wavelength_Error`, `SlitPolyA`, `SlitPolyB`, `SlitPolyC`, `SlitFraction`
- `Wavelength`: [c0, c1, c2] polynomial, wl(x) = c0 + c1*x + c2*x^2, in nm
- `SlitPolyA`: slit shape polynomial [c0, c1, c2] — default `[0, 1, 0]` (identity)
- `SlitPolyB`: slit tilt polynomial [c0, c1, c2] — constant tilt per trace, now populated
- All polynomial arrays have 3 coefficients (quadratic)

## Science reduction

### Setup (`make_reduction_sofs.py`)
- Pairs A+B frames from LMscience.sqlite using greedy matching within each template sequence
- 5239 AB pairs across 16 settings, 414 combined template sequences
- Directory structure: `reduced/{object}_{setting}_{tpl_start}_{pair}/nodd.sof`
- SOF references: `../../raw/{dp_id}.fits`, `../../{setting}_tw.fits`, `../../flats/{setting}/cr2res_cal_flat_Open_master_flat.fits`
- 229 frames unpaired (odd counts or all-same nod position)
- Non-standard templates (raster, spectroastrometry, 12 sequences) excluded
- Some sequences labeled as nodding have only one nod position (Jupiter, Venus, Moon scans) — these fail `cr2res_obs_nodding` (nod throw=0 or unequal A/B). Removed from database or skipped by `make_combined_sofs.py`

### Pairs per setting
- L3244: 9, L3262: 473, L3302: 404, L3340: 358, L3377: 353, L3412: 30, L3426: 92
- M4187: 35, M4211: 520, M4266: 57, M4318: 611, M4368: 2244, M4416: 9, M4461: 35, M4504: 5, M4519: 4

### esorex recipe config
- `cr2res_obs_nodding.rc` — recipe parameter file for nodding reduction
- Generated via `esorex --create-config=cr2res_obs_nodding.rc cr2res_obs_nodding`
- Custom settings: `extract_swath_width=2048`, `extract_height=45`, `extract_oversample=10`
- Usage: `esorex --recipe-config=../../cr2res_obs_nodding.rc cr2res_obs_nodding nodd.sof`
- CLI flags override config file values

### Batch reduction + telluric + wavelength correction
```
ls reduced/*/nodd.sof | parallel -j 4 --bar 'cd {//} && esorex --recipe-config=../../cr2res_obs_nodding.rc cr2res_obs_nodding nodd.sof 2>&1 > esorex.log && cd ../.. && uv run tellcorr.py {//} && uv run wavecorr.py {//}'
```
To skip already-completed dirs, filter before piping to parallel:
```
ls reduced/*/nodd.sof | sed 's|/nodd.sof||' | while read d; do [ -e "$d/cr2res_obs_nodding_extractedA_tellcorr.fits" ] || echo "$d"; done | parallel -j 6 --bar 'cd {} && esorex --recipe-config=../../cr2res_obs_nodding.rc cr2res_obs_nodding nodd.sof 2>&1 > esorex.log && cd ../.. && uv run tellcorr.py {} && uv run wavecorr.py {}'
```

## Telluric correction script (`tellcorr.py`)

### Usage
- `uv run tellcorr.py <reduction_dir>`
- Takes a reduction directory, processes both extractedA and extractedB
- Runs vipere in a temp directory, reconstructs model from residuals, writes corrected FITS + diagnostic plots

### Output files per directory
- `cr2res_obs_nodding_extractedA_tellcorr.fits` — corrected A spectrum
- `cr2res_obs_nodding_extractedB_tellcorr.fits` — corrected B spectrum
- `tellcorrAB_{NN}.png` — one diagnostic plot per spectral order (3 chips together)

### Output FITS columns (per order, per chip extension)
- `XX_01_SPEC` — telluric-corrected spectrum (ADU), `= original / TELLUR`
- `XX_01_ERR` — error (unchanged from pipeline)
- `XX_01_WL` — wavelength in nm (pipeline; updated by wavecorr.py after tellcorr)
- `XX_01_TELLUR` — telluric transmission (0-1, from vipere model / continuum)
- `XX_01_CONT` — fitted continuum polynomial (ADU)
- `CONT * TELLUR` reconstructs the full vipere model in data units
- Orders where vipere failed: SPEC = original, WL = pipeline, TELLUR = NaN, CONT = NaN

### How telluric model is reconstructed
- `model[pixel] = observed[pixel] - residual[pixel]` from vipere `res/` files
- Interpolated to fill pixel gaps within fitted range; NaN outside
- `continuum = polyval(pixel - xcen, norm_coeffs)` from par.dat
- `telluric = model / continuum` (atmospheric transmission at detector resolution, includes IP convolution)
- `corrected = observed / telluric`

### Key parameters
- `-vcut 0` — uses full detector range (no edge trimming)
- `-plot 0` with `MPLBACKEND=Agg` — suppresses all GUI windows
- `--oset` CLI flag to override vipere order range
- `TELLCORR_KEEP_WORKDIR=1` env var to keep temp directory for debugging

### Sidecar files (for wavecorr.py)
- `tellfit_{A,B}.par.dat` — copy of vipere par.dat
- `tellfit_{A,B}_xcen.json` — map of `"chip_order": xcen` for each fitted order

### Diagnostic plots (tellcorrAB)
- Top panel: uncorrected A (blue) and B (orange) spectra with vipere model overlay (black = CONT * TELLUR)
- Bottom panel (narrower): residuals (data - model) for A and B
- Each model is scaled independently per nod position

### Error handling
- Empty CHIP extensions (bad esorex output): detected, bad file deleted, skip gracefully
- Missing orders per chip (e.g. M4211 CHIP1 missing order 8): skipped in plots

## Wavelength correction script (`wavecorr.py`)

### Usage
- `uv run wavecorr.py <reduction_dir>`
- Reads tellfit sidecar files from tellcorr.py, updates WL columns in _tellcorr.fits

### Method
- **Fitted orders**: vipere wavelength polynomial applied directly (Angstrom -> nm)
- **Unfitted orders**: 2D polynomial fit to vipere wavelengths wl(x, order_number) per chip, then evaluated at the unfitted order numbers
- 2D fit: degree 2 in pixel, degree 2 in order number (9 terms), fitted to 20 sample points per fitted order
- Each chip fitted independently
- Diagnostic 3D plot saved as `wavecorr_chip{N}_{A,B}.png` showing surface + data points + order lines (green=fitted, red dashed=interpolated)

### FITS extension access
- Always use `hdul['CHIP{N}.INT1']` not `hdul[N]` — combined reductions may have extra extensions shifting numeric indices

## Trace diagnostic script (`plot_ABtraces.py`)

### Usage
- `uv run plot_ABtraces.py <reduction_dir>`
- Plots `abs(combinedA)` images for all 3 chips with A and B extraction windows overlaid
- Output: `ABtraces.png` in the reduction directory
- Extraction height read from `ESO PRO REC1 PARAM{N}` headers in trace_wave_A.fits
- Traces shown as black/white alternating dashed lines (visible on any background)

## Web app (`webapp.py`)

### Architecture
- FastAPI + Jinja2 templates + HTMX for filtering + Plotly for interactive spectra
- Reads metadata from `LMscience.sqlite` (view `observations` created on first run with `CREATE VIEW IF NOT EXISTS`)
- Serves files from `reduced/` directories (PNGs and tellcorr FITS)
- Run locally: `uv run webapp.py --reload`, deployed via systemd (`crires-lm-webapp.service`)

### URL routing
- `/` — filterable/sortable table of all observations with tellcorr spectra
- `/obs/{dirname}` — observation page (combined template or individual pair)
- `/about` — renders README.md via `markdown` library
- `/api/spectrum/{dirname}` — JSON endpoint returning Plotly figure data
- `/files/{dirname}/{filename}` — serves PNGs and FITS from `reduced/`

### Reverse proxy support
- `--root-path /crires-lm` sets the ASGI root_path, exposed to templates as `{{ base }}`
- All template URLs use `{{ base }}` prefix; empty string when running locally
- Caddy config uses `handle_path` (not `handle`) to strip the prefix before forwarding

### Directory resolution
- Combined template dirs: `{object}_{setting}_{tpl_short}` — mapped 1:1 to tpl_start via `_dir_to_tpl`
- Pair dirs: `{object}_{setting}_{tpl_short}_{N}` — resolved by stripping `_N` suffix
- `_resolve_dirname()` handles both, returns `(tpl_start, base_dirname, pair_num)`

### AB pair display
- `_pair_frames()` does greedy A-B matching (same algorithm as `make_reduction_sofs.py`)
- Pairs shown with alternating background colors, joint hover highlight via JS
- Clicking either row of a pair navigates to `/obs/{dirname}_{N}`
- Pair pages show only their 2 frames, with back-link to combined template page

### Front page table
- Client-side sorting via JS on column headers (text or numeric)
- HTMX-powered filtering (target, setting, programme, search)
- Only shows observations with `_tellcorr.fits` present
