import pickle
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import wiener

from superreg.fouriershift import Register
from superreg.util.tester import BiasTest
import superreg.util.makedata as md

rng = np.random.RandomState(148509289)
delta = rng.rand(2)*2

mpl.rcParams['font.size'] = 16.
mpl.rcParams['axes.labelsize'] = 14.
mpl.rcParams['axes.titlesize'] = 20.
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['lines.linewidth'] = 1.5

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

def crb(I, sigma):
    ly, lx = I.shape
    Ik = np.fft.fftn(I, norm='ortho')  # Unitary norm!
    ky = np.fft.fftfreq(ly, d=1./(2*np.pi))[:,None]
    kx = np.fft.fftfreq(lx, d=1./(2*np.pi))
    return sigma **2 / np.sum(ky**2 * Ik * Ik.conj()).real

#dirname = os.path.realpath("../N_1000-2018-02-12")
dirname = os.path.realpath("results/N_1000-2018-03-20")
dirname_sr = os.path.realpath("results/N_500-2018-04-07")

directory = os.path.realpath("results/N_1000-deg_13-2018-04-12")
expt = loadresults(directory)
marg = expt['periodic-shift.pkl']
superreg = expt['periodic-superreg.pkl']
noises = marg['noises']
img = marg['datakwargs']['random']['img']

margcolor = "#f3899a"
srcolor = "#007f6f"

#==========================================================================
#  Figure demostrating marginal likelihood
#==========================================================================

demoshift = np.array(img.shape)//3
demodata = md.fakedata(0.07, shifts=[-demoshift], L=img.shape[0],
                       offset=np.zeros(2), img=img, mirror=False)
Ly, Lx = img.shape
packed = np.zeros(np.concatenate([np.array(img.shape)+demoshift, [4]]))

fig0, ax0 = plt.subplots(figsize=(4.3, 4.3))
fig1, ax1 = plt.subplots(figsize=(4.3, 4.3))
fig2, ax2 = plt.subplots(figsize=(4.5, 3.))

dd = mpl.cm.bone_r(mpl.colors.Normalize()(img))
dd0 = mpl.cm.bone_r(mpl.colors.Normalize()(demodata[0]))
dd1 = mpl.cm.bone_r(mpl.colors.Normalize()(demodata[1]))
alpha = 0.5
dd0[:,:,3] = alpha
dd1[:,:,3] = alpha
packed[:Ly,:Lx,:,] = dd1.copy()
sy, sx = demoshift
packed[sy:sy+Ly,sx:sx+Lx,:,] = dd0.copy()
packed[sy:-sy, sx:-sx,3] = 1
ax0.imshow(packed)
ax0.arrow(0,0,*demoshift, width=3, length_includes_head=True, 
          facecolor=margcolor, zorder=2)
bbox = dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.05',
            edgecolor='none')
ax0.text(*(demoshift/2. + np.array([5,0])), r"$\mathbf{\Delta}$",
           fontdict={'fontsize': 20, 'color': margcolor}, zorder=2,
            bbox=bbox)
ax0.axis('off')
#ax[0].text(0, Ly+sy+18, "(a)", fontdict={"color": margcolor})
ax0.set_title("Standard Fourier Shift")
ax0.text(sx/8., Ly+7*sy/8, "(a)", bbox=bbox, fontsize=18)

#==========================================================================
#  Figure demostrating maximum likelihood
#==========================================================================

packedsr = np.zeros(np.concatenate([np.array(img.shape)+demoshift, [4]]))
dd0[:,:,3] = 1.
dd1[:,:,3] = 1.
packedsr[:Ly,:Lx,:] = dd1
packedsr[-Ly:,-Lx:,:] = dd0
ax1.imshow(packedsr, zorder=0)
fimg = wiener(img, 6)
packedimg = np.zeros_like(packed[:,:,0])
# TODO: correct the orientation of the contour image!
packedimg[sy:,sx:] = fimg
packedimg[:sy,:sx] = fimg[-sy:,-sx:]
packedimg[:sy,sx:] = fimg[-sy:,:]
packedimg[sy:,:sx] = fimg[:,-sx:]

iy = np.arange(packedimg.shape[0])
ix = np.arange(packedimg.shape[1])
ixg, iyg = np.meshgrid(ix, iy)
ax1.contour(ixg, iyg, packedimg, cmap='Greens', linewidths=1.5, zorder=1,
              alpha=0.8)
ax1.arrow(0,0,*demoshift, width=3, length_includes_head=True, 
            facecolor= srcolor, zorder=2)
t = ax1.text(*(demoshift/2. + np.array([5,0])), r"$\mathbf{\Delta}$",
           fontdict={'fontsize': 20, 'color': srcolor}, zorder=2,
            bbox=dict(facecolor='white', alpha=0.7,
                      boxstyle='round,pad=0.05', edgecolor='none'))

ax1.axis('off')
rect = mpl.patches.Rectangle((sy,sx), Lx, Ly, edgecolor='white',
                             facecolor='none', linestyle='dashed',
                             linewidth=2, zorder=2)
ax1.add_patch(rect)
#ax[1].text(0, Ly+sy+18, "(b)", fontdict={"color": srcolor})
ax1.set_title("Super Registration")
ax1.text(sx/8, Ly+7*sy/8., "(b)", bbox=bbox, fontsize=18)

#==========================================================================
#  Figure summarizing errors and bias
#==========================================================================

merr = ax2.plot(noises, marg['results']['bias_std'], 'o', c=margcolor,
                  zorder=3)
#mcrb = ax[2].plot(noises, marg['results']['err'], '-', c=margcolor, zorder=1)
mcrb = ax2.fill_between(noises, marg['results']['err'], y2=0.,
                          color=margcolor, zorder=2, alpha=0.4)
theory_error = np.array([np.sqrt(predictvariance(img, n)) for n in noises])
mcalc = ax2.plot(noises, theory_error, linestyle=':', c='k', zorder=4)

srerr = ax2.scatter(noises, superreg['results']['bias_std'], marker='+',
              c=srcolor, zorder=4)

#srcrb = ax[2].plot(noises, superreg['results']['err'], '-', c=srcolor, zorder=3)
srcrb = ax2.fill_between(noises, superreg['results']['err'], y2=0.,
                           color=srcolor, zorder=1, alpha=0.3)

ax2.set_xlabel("Noise $\sigma$")
labels = ("Fourier Shift (FS)", "FS CRB",
          "Theory", "Super Registration (SR)",
          "SR CRB")
lines = (merr[0], mcrb, mcalc[0], srerr, srcrb)
#lines = (merr[0], mcrb[0], mcalc[0], srerr, srcrb[0])
ax2.legend(lines, labels, loc='upper left', bbox_to_anchor=(-.01, 1.01))
#ax2.set_title("$\Delta$ Error Comparison")
ax2.set_xlim([0, 0.1])
ax2.set_ylim([0, 0.2])
ax2.set_xticks([0., 0.025, 0.05, 0.075, 0.1])
ax2.set_xticklabels(["0", "0.025", "0.05", "0.075", "0.1"])
ax2.set_yticks([0., 0.05, 0.1, 0.15, 0.2])
ax2.set_yticklabels(["0", "0.05", "0.1", "0.15", "0.2"])
ax2.set_ylabel('$\Delta$ Error (pixels)')

#square(ax2)
plt.tight_layout()


#p = list(ax1.get_position().bounds)
#p[0] -= 0.015
#ax[1].set_position(p)

plt.show()
