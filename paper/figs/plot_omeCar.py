import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Roboto'

FITS_PATH = 'ome_Car_L3262_2023-12-24_07335.fits'
ORDER = '03'
CHIPS = [1, 2, 3]

fi = fits.open(FITS_PATH)

fig, axes = plt.subplots(3, 1, figsize=(7, 5))

ytop = 40000
scale = 1.
shift = 12000

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

    ax.axvline(3.69320, color='grey', lw=0.5, ls=':')  # Hu 18-6

    wmin, wmax = np.nanmin(w) / 1000, np.nanmax(w) / 1000
    ax.set_xlim(wmin, wmax)
    ax.set_ylim(-500, ytop)
    ax.set_ylabel('Flux [ADU]')

    ax.text(0.01, 0.95, f'CHIP{chip}', transform=ax.transAxes,
            va='top', ha='left', fontsize=9)
    ax.tick_params(direction='in', top=True, right=True)

axes[0].set_title(r'$\omega$ Car — setting L3262 — order 03')
axes[0].legend(loc='lower left', framealpha=0, fontsize=8)
axes[-1].set_xlabel(r'Wavelength [$\mu$m]')

fig.subplots_adjust(hspace=0.18)

fig.savefig('omeCar_Hu18.pdf', bbox_inches='tight')
fig.savefig('omeCar_Hu18.png', bbox_inches='tight', dpi=150)
