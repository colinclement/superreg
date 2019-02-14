import pickle
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.signal import wiener

from super_reg.twod.fouriershift import Register
from super_reg.util.tester import BiasTest
import super_reg.util.makedata as md

rng = np.random.RandomState(148509289)
delta = rng.rand(2)*2

mpl.rcParams['font.size'] = 16.
mpl.rcParams['axes.labelsize'] = 14.
mpl.rcParams['axes.titlesize'] = 16.
mpl.rcParams['axes.titlepad'] = 10.
mpl.rcParams['legend.fontsize'] = 12

def loadresults(directory):
    files = os.listdir(directory)
    files.remove('summary.pdf')
    data = {}
    for i, f in enumerate(files):
        with open(os.path.join(directory, f), "rb") as infile:
            res = pickle.load(infile)
            noises = pickle.load(infile)
            datakwargs = pickle.load(infile)
            mask_kwargs = pickle.load(infile)
            alldata = pickle.load(infile)
            shifts = pickle.load(infile)
            img = pickle.load(infile)
        data[f] = {'results': res, 'noises': noises,
                   'datakwargs': datakwargs, 'mask_kwargs': mask_kwargs,
                   'alldata': alldata, 'shifts': shifts, 'img': img}
    return data

def square(ax):
    axis = ax.axis()
    ax.set_aspect(np.diff(axis[:2])/np.diff(axis[2:]))

# NOTE: I only plot shift bias/error in the y-direction

def predictvariance(I, sigma):
    ly, lx = I.shape
    n = ly*lx
    Ik = np.fft.fftn(I, norm='ortho')  # Unitary norm!
    ky = np.fft.fftfreq(ly, d=1./(2*np.pi))[:,None]
    kx = np.fft.fftfreq(lx, d=1./(2*np.pi))
    denom = np.sum(kx**2 * Ik * Ik.conj()).real
    sumks = n * np.pi**2/3. # np.sum(ky**2) * lx
    return 2*sigma**2/denom + sigma**4 * sumks / denom**2

def crb(img, sigma, deg=None):
    Ly, Lx = img.shape
    degy = deg if deg is not None else Ly
    degx = deg if deg is not None else Lx
    ky = np.fft.fftfreq(Ly, d=1./(2*np.pi))
    ky[degy:] = 0.
    kx = np.fft.fftfreq(Lx, d=1./(2*np.pi))
    kx[degx:] = 0.
    Ik = np.fft.fftn(img, norm='ortho')
    Ik2 = Ik*Ik.conj()
    return sigma**2/np.sum(ky**2*Ik2).real, sigma**2/np.sum(kx**2*Ik2).real
 
dirname = "results/complexity-N_2000-deglist_62-noise_0.075-2018-04-26"
directory  = os.path.realpath(dirname)
srcolor = "#007f6f"

expt = loadresults(directory)
marg = expt['periodic-superreg.pkl']
results = marg['results']
deglist = np.array(range(2, 64))
img = marg['img']
truecrb = np.array([crb(img, 0.075, d) for d in deglist])

fig, axe = plt.subplots(figsize=(5.7, 3.9))
axe2 = axe.twinx()

evd = np.array(results['evidence'])
bestdeg = deglist[np.argmax(results['evidence'])]
scale = 1E4
evdalpha = 1.
evdplot = axe2.plot(deglist, -evd/scale, c='k', zorder=1, alpha=evdalpha)
f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
axe2.yaxis.set_major_formatter(mtick.FuncFormatter(g))
limits = [1.43, 3.900]
axe2.set_ylim(limits)
mevdline = axe2.axvline(bestdeg, 0,
                        (-max(evd)/scale-limits[0])/(np.diff(limits)[0]), c='k',
                        ls=':', zorder=3, alpha=evdalpha)
axe2.set_ylabel("Negative log evidence ($10^4$)")
axe2.set_xticks([0, bestdeg, 30., 45., 60.])
axe2.set_xticklabels(['0', '$\lambda^\star$', '30', '45', '60'])
#axe2.legend(loc='upper right')

errp = axe.scatter(deglist, results['bias_std'], c=srcolor, marker='+',
                   zorder=2) 
crbline = axe.plot(deglist, results['err'], c=srcolor, label="$\Delta$ CRB",
                   zorder=2)
merrline = axe.axhline(min(results['bias_std']), 0, bestdeg/60.,
                       c=srcolor, ls=':', zorder=1)
truecrbline = axe.plot(deglist, np.sqrt(truecrb[:,1]), c=srcolor, ls='dashed',
                       zorder=1)
axe.set_xlabel("Complexity of image model $\lambda$")
axe.set_ylabel("$\Delta$ error ($\sigma_\Delta$)", labelpad=-20)
axe.set_xlim([0, 64])
axe.set_ylim([0., .205])
minerr = min(results['bias_std'])
axe.set_yticks([0., 0.05, minerr, .1, .15, .2],)
axe.set_yticklabels(('0', '', 'min($\sigma_\Delta$)', '0.1', '', '0.2'))

#labels = ("Evidence $-\log p(\phi,\psi)$", "Max. evidence complexity",
#          "Measured error", "Estimated CRB", "True CRB", "Minimum error")
labels = ("Evidence $-\log p(\phi,\psi)$",
          "Measured error", "Estimated CRB", "True CRB")
#axe.legend((evdplot[0], mevdline, errp, crbline[0], truecrbline[0], 
#            merrline), labels, loc=(.1, .31))
axe.legend((evdplot[0], errp, crbline[0], truecrbline[0]),
           labels, loc=(.15, .41))
axe.set_title("Maximum evidence minimizes error ($\sigma=0.075$)",
              fontdict=dict(verticalalignment='baseline'))
plt.tight_layout()
fig.set_size_inches([6.41, 3.3])
 
plt.show()
