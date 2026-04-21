import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Roboto'

FITS_PATH = 'bet_Ori_M4368_2024-12-09_02033.fits'
ORDER = '06'
CHIP = 2  # only CHIP2 contains Brα 4.052 μm

fi = fits.open(FITS_PATH)

fig, ax = plt.subplots(1, 1, figsize=(7, 2.5))

ytop = 45000
scale = 1.
shift = 10000

d = fi[CHIP].data
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

ax.axvline(4.05226, color='grey', lw=0.5, ls=':')

wmin, wmax = np.nanmin(w) / 1000, np.nanmax(w) / 1000
ax.set_xlim(wmin, wmax)
ax.set_ylim(-500, ytop)
ax.set_ylabel('Flux [ADU]')
ax.set_xlabel(r'Wavelength [$\mu$m]')

ax.text(0.01, 0.95, f'CHIP{CHIP}', transform=ax.transAxes,
        va='top', ha='left', fontsize=9)
ax.tick_params(direction='in', top=True, right=True)

ax.set_title(r'$\beta$ Ori (Rigel) — setting M4368 — order 06 — Br$\alpha$')
ax.legend(loc='lower left', framealpha=0, fontsize=8)

fig.savefig('betOri_Bra.pdf', bbox_inches='tight')
fig.savefig('betOri_Bra.png', bbox_inches='tight', dpi=150)
