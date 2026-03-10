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

All 5239 AB pairs and their combined per-template spectra are reduced with `esorex cr2res_obs_nodding` using the corrected tracing tables (with slit tilt and updated wavelengths). Then `tellcorr.py` runs vipere on each extracted spectrum to fit and divide out the telluric absorption, updating the wavelength solution simultaneously. For orders where vipere cannot fit (no telluric features, CO2-saturated regions), the wavelength correction is interpolated from a 2D polynomial fitted to the orders that do have vipere solutions.

The reduced and telluric-corrected spectra are browsable and downloadable via a web app (link TBD).

## Data scale

- 16 wavelength settings (L3244--L3426, M4187--M4519), 5--7 echelle orders per chip, 3 chips
- 11,135 raw science frames from the ESO archive, forming 5239 AB nod pairs across 442 observing sequences
- Combined per-template spectra for all sequences

## Repository structure

```
*.py                   scripts (fetch, reduce, telluric correct, webapp)
*_tw.fits              tracing/wavelength tables per setting (calibration products)
cr2res_obs_nodding.rc  esorex recipe configuration
sof/                   SOF files for flat reduction
flats/sof/             SOF files for flat reduction (deep flats)
tellurics/             telluric standard reductions, tilt + wavecal scripts
templates/             HTML templates for the inspection webapp
```

Raw data, reduced products, and FITS outputs are not included (too large); they live on disk and can be regenerated from the scripts + ESO archive.

## Requirements

- [ESO esorex](https://www.eso.org/sci/software/cpl/esorex.html) with the `cr2res` pipeline
- [vipere](https://git.astro.lavail.net/alexis/vipere) (with L/M band atmospheric models)
- Python 3.10+, managed via [uv](https://github.com/astral-sh/uv); all scripts have [PEP 723](https://peps.python.org/pep-0723/) inline metadata so `uv run script.py` pulls in the right dependencies automatically
