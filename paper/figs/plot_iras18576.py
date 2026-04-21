import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Roboto'

FITS_PATH = 'IRAS_18576+0341_L3340_2022-08-09_04324.fits'
ORDER = '04'
CHIPS = [1, 2, 3]

fi = fits.open(FITS_PATH)

fig, axes = plt.subplots(3, 1, figsize=(7, 5))

ytop = 10000
scale = 1.
shift = 3000

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

    ax.axvline(3.54434, color='grey', lw=0.5, ls=':')  # Hu 22-6

    wmin, wmax = np.nanmin(w) / 1000, np.nanmax(w) / 1000
    ax.set_xlim(wmin, wmax)
    ax.set_ylim(-100, ytop)
    ax.set_ylabel('Flux [ADU]')

    ax.text(0.01, 0.95, f'CHIP{chip}', transform=ax.transAxes,
            va='top', ha='left', fontsize=9)
    ax.tick_params(direction='in', top=True, right=True)

axes[0].set_title(r'IRAS 18576+0341 — setting L3340 — order 04 — Hu 22$\to$6')
axes[0].legend(loc='lower left', framealpha=0, fontsize=8)
axes[-1].set_xlabel(r'Wavelength [$\mu$m]')

fig.subplots_adjust(hspace=0.18)

fig.savefig('iras18576_Hu22.pdf', bbox_inches='tight')
fig.savefig('iras18576_Hu22.png', bbox_inches='tight', dpi=150)
