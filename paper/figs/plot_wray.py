import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Roboto'

FITS_PATH = 'WRAY_15-1676_M4318_2021-09-15_23331.fits'
ORDER = '04'
CHIPS = [1, 2, 3]

fi = fits.open(FITS_PATH)

fig, axes = plt.subplots(3, 1, figsize=(7, 5))

ytop = 8500
scale = 1.
shift = 2500

for ax, chip in zip(axes, CHIPS):
    d = fi[chip].data
    w = d[f'{ORDER}_01_WL']
    s = d[f'{ORDER}_01_SPEC']
    c = d[f'{ORDER}_01_CONT']
    t = d[f'{ORDER}_01_TELLUR']

    observed = s * t
    model = c * t
    corrected = np.where(t > 0.3, s, np.nan)

    ax.plot(w / 1000, observed, color='k', lw=1, label='observed')
    ax.plot(w / 1000, model, color='r', lw=0.7, alpha=0.8, label='telluric model')
    ax.plot(w / 1000, corrected * scale + shift, color='C0', lw=0.5, alpha=0.8,
            label=f'{shift} + obs / tell')

    wmin, wmax = np.nanmin(w) / 1000, np.nanmax(w) / 1000
    ax.set_xlim(wmin, wmax)
    ax.set_ylim(-100, ytop)
    ax.set_ylabel('Flux [ADU]')

    ax.text(0.01, 0.95, f'CHIP{chip}', transform=ax.transAxes,
            va='top', ha='left', fontsize=9)
    ax.tick_params(direction='in', top=True, right=True)

axes[0].set_title(r'WRAY 15-1676 — setting M4318 — order 04')
axes[0].legend(loc='lower left', framealpha=0, fontsize=8)
axes[-1].set_xlabel(r'Wavelength [$\mu$m]')

fig.subplots_adjust(hspace=0.18)

fig.savefig('wray_CO.pdf', bbox_inches='tight')
fig.savefig('wray_CO.png', bbox_inches='tight', dpi=150)
