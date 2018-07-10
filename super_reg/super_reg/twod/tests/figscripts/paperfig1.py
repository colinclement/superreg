import pickle
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import wiener

from super_reg.twod.fouriershift import Register
from super_reg.util.tester import BiasTest
import super_reg.util.makedata as md

rng = np.random.RandomState(148509289)
delta = rng.rand(2)*2

mpl.rcParams['font.size'] = 16.
mpl.rcParams['axes.labelsize'] = 14.
mpl.rcParams['axes.titlesize'] = 20.
mpl.rcParams['legend.fontsize'] = 10

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
dirname = os.path.realpath("../N_1000-2018-03-20")
dirname_sr = os.path.realpath("../N_500-2018-04-07")

directory = os.path.realpath("../N_1000-deg_13-2018-04-12")
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

fig, ax = plt.subplots(1, 3, figsize=(10, 4.6))
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
ax[0].imshow(packed)
ax[0].arrow(0,0,*demoshift, width=3, length_includes_head=True, 
            facecolor=margcolor, zorder=2)
ax[0].text(*(demoshift/2. + np.array([5,0])), r"$\mathbf{\Delta}$",
           fontdict={'fontsize': 20, 'color': margcolor}, zorder=2,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.05',
                      edgecolor='none'))
ax[0].axis('off')
#ax[0].text(0, Ly+sy+18, "(a)", fontdict={"color": margcolor})
ax[0].set_title("Marginal ML")

#==========================================================================
#  Figure demostrating maximum likelihood
#==========================================================================

packedsr = np.zeros(np.concatenate([np.array(img.shape)+demoshift, [4]]))
dd0[:,:,3] = 1.
dd1[:,:,3] = 1.
packedsr[:Ly,:Lx,:] = dd1
packedsr[-Ly:,-Lx:,:] = dd0
ax[1].imshow(packedsr, zorder=0)
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
ax[1].contour(ixg, iyg, packedimg, cmap='Greens', linewidths=1.5, zorder=1,
              alpha=0.8)
ax[1].arrow(0,0,*demoshift, width=3, length_includes_head=True, 
            facecolor= srcolor, zorder=2)
t = ax[1].text(*(demoshift/2. + np.array([5,0])), r"$\mathbf{\Delta}$",
           fontdict={'fontsize': 20, 'color': srcolor}, zorder=2,
            bbox=dict(facecolor='white', alpha=0.7,
                      boxstyle='round,pad=0.05', edgecolor='none'))

ax[1].axis('off')
rect = mpl.patches.Rectangle((sy,sx), Lx, Ly, edgecolor='white',
                             facecolor='none', linestyle='dashed',
                             linewidth=2, zorder=2)
ax[1].add_patch(rect)
#ax[1].text(0, Ly+sy+18, "(b)", fontdict={"color": srcolor})
ax[1].set_title("Super Registration")

#==========================================================================
#  Figure summarizing errors and bias
#==========================================================================

merr = ax[2].plot(noises, marg['results']['bias_std'], 'o', c=margcolor,
                  zorder=1)
mcrb = ax[2].plot(noises, marg['results']['err'], '-', c=margcolor, zorder=1)
theory_error = np.array([np.sqrt(predictvariance(img, n)) for n in noises])
mcalc = ax[2].plot(noises, theory_error, linestyle=':', c='k', zorder=1)

srerr = ax[2].scatter(noises, superreg['results']['bias_std'], marker='+',
              c=srcolor, zorder=3)

srcrb = ax[2].plot(noises, superreg['results']['err'], '-', c=srcolor, zorder=3)

ax[2].set_xlabel("Noise $\sigma$")
labels = ("Marginal ML Error", "Marginal ML CRB",
          "Corrected ML Error", "SR Error",
          "SR CRB")
lines = (merr[0], mcrb[0], mcalc[0], srerr, srcrb[0])
ax[2].legend(lines, labels, loc='upper left')
ax[2].set_title("$\Delta$ Error Comparison")

square(ax[2])

p = list(ax[1].get_position().bounds)
p[0] -= 0.0225
ax[1].set_position(p)

plt.show()
