# CRIRES-LM

Bulk reduction of all public CRIRES+ L-band (2.8--4.2 um) and M-band (4.4--5.5 um) science data from the ESO archive (September 2021 through March 2025), with improved slit tilt calibration and per-observation telluric correction using [vipere](https://git.astro.lavail.net/alexis/vipere) (a fork of [viper](https://github.com/mzechmeister/viper)).

- **Paper:** [arXiv:2604.24466](https://arxiv.org/abs/2604.24466) — Marquart & Lavail (2026)
- **Data browser:** <https://www.astro.uu.se/crires-lm/>
- **Data archive (Zenodo):** [10.5281/zenodo.19675664](https://doi.org/10.5281/zenodo.19675664)
- **Code snapshot (Zenodo):** [10.5281/zenodo.19754514](https://doi.org/10.5281/zenodo.19754514)

The paper describes the calibration approach, data products, and known limitations. This repository contains the scripts that produce the archive and the corrected `*_tw.fits` tracing tables (slit tilt + improved wavelength solutions) that can be used as drop-in inputs for custom `cr2res` reductions.

## Citation

If you use these data or scripts, please cite:

```bibtex
@article{marquart2026crireslm,
  author        = {Marquart, Thomas and Lavail, Alexis},
  title         = {An archive of reduced and telluric-corrected CRIRES$^+$
                   L- and M-band spectra with slit-tilt and wavelength calibrations},
  journal       = {arXiv e-prints},
  year          = {2026},
  eprint        = {2604.24466},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.IM},
  url           = {https://arxiv.org/abs/2604.24466}
}

@misc{crires_lm_data,
  author       = {Marquart, Thomas and Lavail, Alexis},
  title        = {{CRIRES+ L/M-band reprocessed spectra}},
  howpublished = {Zenodo},
  year         = {2026},
  doi          = {10.5281/zenodo.19675664},
  url          = {https://doi.org/10.5281/zenodo.19675664}
}
```

## Repository structure

```
*.py                   scripts (fetch, reduce, telluric correct, wavecal, compare, webapp)
*_tw.fits              tracing/wavelength tables per setting (calibration products, aggregate-corrected)
cr2res_obs_nodding.rc  esorex recipe config for nodding
cr2res_cal_flat.rc     esorex recipe config for flats
flats/                 flat field data: {setting}_{date}/ subdirs with SOFs and output
tellurics/             telluric standard reductions, tilt + wavecal scripts
templates/             HTML templates for the inspection webapp
paper/                 LaTeX source of the paper
```

Raw data, reduced products, and FITS outputs are not included (too large); they live on disk and can be regenerated from the scripts + ESO archive.

## Requirements

- [ESO esorex](https://www.eso.org/sci/software/cpl/esorex.html) with the `cr2res` pipeline
- [vipere](https://git.astro.lavail.net/alexis/vipere) with L/M band atmospheric models ([stdAtmos_L.fits](https://neon.physics.uu.se/crires/stdAtmos_L.fits), [stdAtmos_M.fits](https://neon.physics.uu.se/crires/stdAtmos_M.fits)) placed in `vipere/lib/atmos/`
- Python 3.10+, managed via [uv](https://github.com/astral-sh/uv); all scripts have [PEP 723](https://peps.python.org/pep-0723/) inline metadata so `uv run script.py` pulls in the right dependencies automatically
