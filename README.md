# CRIRES-LM

Bulk reduction of all public CRIRES+ L-band (2.8--4.2 um) and M-band (4.2--5.5 um) science data from the ESO archive (2021-01 to 2025-02), with improved slit tilt calibration and per-observation telluric correction using [vipere](https://git.astro.lavail.net/alexis/vipere) (a fork of [viper](https://github.com/mzechmeister/viper)).

## Approach

The standard CRIRES+ pipeline (`cr2res`) does not account for slit tilt in L/M bands, and ships no telluric correction. We use vipere in two stages:

### Stage 1: Calibration from telluric standards

For each of the 16 wavelength settings, we reduce one telluric standard star (hot B/O-type) observed at nod positions A and B.
Vipere fits a forward model of telluric absorption + wavelength solution to each nod position independently.
The A vs B wavelength difference at each pixel directly measures the slit tilt, since the two nod positions sample different spatial positions on the slit.

From these measurements:

- **Slit tilt**: fit a linear tilt(wavelength) relation per band, interpolating across the CO2 gap at 4.2--4.5 um where no telluric fit is possible. Written into the `SlitPolyB` column of the tracing tables (`*_tw.fits`).
- **Wavelength calibration**: vipere wavelength polynomials replace the pipeline solution for 72% of traces; the remainder keep pipeline wavelengths (mostly orders with weak/no telluric features).
- **Nod throw**: actual throw measured from spatial profiles (2--3% larger than commanded).

### Stage 2: Science reduction with telluric correction

Before extraction, the trace positions are adjusted per observation to account for instrument flexure: `adjust_traces.py` cross-correlates the spatial profile of each raw frame against the reference trace boundaries to measure the Y-shift (typically 5--10 pixels, up to ~50 in rare cases) and writes a corrected tracing table into each reduction directory.

The pipeline-provided tracing tables (`*_tw.fits`) have been cleaned of spurious edge traces that the pipeline's tracing algorithm occasionally creates for partial orders at detector boundaries. These duplicate traces poison the `SlitFraction` metadata, preventing extraction of valid orders (affected settings: M4211, M4461, M4504).

All 5237 AB pairs and their combined per-template spectra are then reduced with `esorex cr2res_obs_nodding` using these adjusted tracing tables (with slit tilt and updated wavelengths) and the nearest-in-time flat field and blaze function. Then `tellcorr.py` runs vipere on each extracted spectrum to fit and divide out the telluric absorption, followed by `wavecorr.py` which updates the wavelength scale: fitted orders get the vipere wavelength polynomial directly, while unfitted orders (no telluric features, CO2-saturated regions) receive a velocity correction interpolated from a linear fit to the vipere corrections across all three chips.

The reduced and telluric-corrected spectra are browsable and downloadable via a [web app](https://neon.physics.uu.se/crires-lm/).

## Web app

The data browser shows all observations that have telluric-corrected spectra. The front page table is filterable by target, setting, and programme, and sortable by clicking column headers.

Each observation page shows the combined per-template spectrum (interactive Plotly plot), diagnostic plots, and download links for the `_tellcorr.fits` files. The raw frames table groups frames into AB pairs; clicking a pair navigates to the individual pair reduction, which has its own spectrum, plots, and downloads.

## Data scale

- 16 wavelength settings (L3244--L3426, M4187--M4519), 5--7 echelle orders per chip, 3 chips
- 11,135 raw science frames from the ESO archive, forming 5237 AB nod pairs across 412 observing sequences
- 233 flat field epochs across 16 settings, each matched to the nearest-in-time science observation
- Combined per-template spectra for all sequences

## Repository structure

```
*.py                   scripts (fetch, reduce, telluric correct, wavecal, webapp)
*_tw.fits              tracing/wavelength tables per setting (calibration products)
cr2res_obs_nodding.rc  esorex recipe config for nodding
cr2res_cal_flat.rc     esorex recipe config for flats
flats/                 flat field data: {setting}_{date}/ subdirs with SOFs and output
tellurics/             telluric standard reductions, tilt + wavecal scripts
templates/             HTML templates for the inspection webapp
```

Raw data, reduced products, and FITS outputs are not included (too large); they live on disk and can be regenerated from the scripts + ESO archive.

## Requirements

- [ESO esorex](https://www.eso.org/sci/software/cpl/esorex.html) with the `cr2res` pipeline
- [vipere](https://git.astro.lavail.net/alexis/vipere) (with L/M band atmospheric models)
- Python 3.10+, managed via [uv](https://github.com/astral-sh/uv); all scripts have [PEP 723](https://peps.python.org/pep-0723/) inline metadata so `uv run script.py` pulls in the right dependencies automatically
