import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mog import mog

os.sys.path.append('../')
import util
import twoploops

mpl.rcParams['image.cmap'] = 'Greys'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['lines.linewidth'] = 1.5

def square(ax):
    axis = ax.axis()
    ax.set_aspect(np.diff(axis[:2])/np.diff(axis[2:]))

dataloc = 'data/2018-09-09-twop-M_1000-N_8-Nsigma_20-L_64.npz'
# Looks like a=3 has the best variance without large bias
cdataloc = 'data/2018-09-10-twop-coarsen-M_1000-N_8-Nsigma_20-L_64.npz'
data = np.load(dataloc)
cdata = np.load(cdataloc)

cresults=cdata['results']

M=data['M']
N=data['N']
Nsigma=data['Nsigma']
L=data['L']
shifts=data['shifts']
s0=data['s0']
pos=data['pos']
widths=data['widths']
params=data['params']
model=data['model']
sigmalist=data['sigmalist']
# sigma, (sr, fs), noise, p, sigmap
results=data['results']
truep=data['truep'][0]
pnames=data['pnames']

def fisherinformation(pnames=pnames, params=params, truep=truep, model=model, sigma=0.1):
    tmog = twoploops.makemog(model[0], sigma, *params)
    j = tmog.lnprob_jacobian(pnames[:-1], truep[:-1])
    return np.array((j.T.dot(j)).todense().diagonal())[0]

def reorder(p):
    ry = p[2] - p[6]
    rx = p[1] - p[5]
    return np.array([ry, rx, p[3], p[7], p[0], p[1]])

def fixnonsense(results):
    srdata = []
    fsdata = []
    for sig in results:
        srdata.append([s for s in sig[0,:,0]])
        fsdata.append([s for s in sig[1,:,0]])
    return np.array(srdata), np.array(fsdata)

def histograms(results=results, truep=truep, pnames=pnames, sigmalist=sigmalist,
               n=-1):
    srdata, fsdata = fixnonsense(results)
    kw = dict(bins=100, histtype='step', density=True)
    fig, axes = plt.subplots(2, 5)
    ax = axes.flat[0]
    ax.hist(srdata[n,:,2]-srdata[n,:,6], label='SR', **kw)
    ax.hist(fsdata[n,:,2]-fsdata[n,:,6], label='FS', **kw)
    ax.axvline(truep[2]-truep[6], c='k', label="Truth")
    ax.set_title("$y_0-y_1$")
    ax = axes.flat[1]
    ax.hist(srdata[n,:,1]-srdata[n,:,5], label='SR', **kw)
    ax.hist(fsdata[n,:,1]-fsdata[n,:,5], label='FS', **kw)
    ax.axvline(truep[1]-truep[5], c='k', label="Truth")
    ax.set_title("$x_0-x_1$")
    for i, ax in enumerate(axes.flat[2:]):  # don't plot error estimate
        ax.hist(srdata[n,:,i], label='SR', **kw)
        ax.hist(fsdata[n,:,i], label='FS', **kw)
        ax.axvline(truep[i], c='k', label="Truth")
        ax.set_title(pnames[i])
    plt.legend()
    fig.suptitle("$\sigma$={}".format(sigmalist[n]))
    plt.show()
    return fig, axes

def meanvar(results):
    srm, fsm = [], []
    srv, fsv = [], []
    for sig in results:
        # fix numpy array mess
        srdata = np.array([s for s in sig[0,:,0]])
        fsdata = np.array([s for s in sig[1,:,0]])

        sr_ry = srdata[:,2] - srdata[:,6]
        sr_rx = srdata[:,1] - srdata[:,5]
        fs_ry = fsdata[:,2] - fsdata[:,6]
        fs_rx = fsdata[:,1] - fsdata[:,5]

        # ry, rx sigma0, sigma1, amp0, amp1
        srhash = [sr_ry, sr_rx, srdata[:,3], srdata[:,7], srdata[:,0],
                  srdata[:,1]]
        fshash = [fs_ry, fs_rx, fsdata[:,3], fsdata[:,7], fsdata[:,0],
                  fsdata[:,1]]

        srm.append(np.mean(srhash, 1))
        srv.append(np.std(srhash, 1)**2)
        fsm.append(np.mean(fshash, 1))
        fsv.append(np.std(fshash, 1)**2)
    return list(map(np.array, (srm, srv, fsm, fsv)))

from superreg.chebseries import SuperRegistration
from superreg.fouriershift import Register
    
def paperplot(results=results, model=model, sigmalist=sigmalist, sigma=0.1,
              param=6, N=N, truep=truep, cresults=cresults):
    fig, axes = plt.subplots(1, 2, figsize=(6.78, 3.))
    margcolor = "#f3899a"
    srcolor = "#007f6f"
    data = model + sigma * np.random.randn(*model.shape)
    fim = fisherinformation()[param]

    axes[0].text(-3., -3., "(a)", fontsize=12)
    axes[0].matshow(data[0])
    axes[0].axis('off')
    axes[0].set_title("Model $I$ with $\sigma$={:.1f}".format(sigma),
                      pad=-3.)

    srdata, fsdata = fixnonsense(results)
    axes[1].plot(sigmalist, sigmalist/np.sqrt(N*fim), ':', c='k', label='CRB')
    axes[1].plot(sigmalist, srdata.std(1)[:,param], c=srcolor, 
              label="Super Registration")
    axes[1].plot(sigmalist, fsdata.std(1)[:,param], c=margcolor, 
              label="FS reconstruction")
    axes[1].plot(sigmalist, cresults[:,:,2,param].std(1), ':', c=margcolor, 
                 label="FS Coarsened")
    axes[1].text(-.04, 1.48, "(b)", fontsize=12)

    axes[1].set_title("Precision of Position Inference", pad=5.)
    axes[1].set_xlabel("Noise $\sigma$", labelpad=-1.)
    axes[1].set_ylabel("Average $y$-position error")
    axes[1].set_xlim([sigmalist.min(), sigmalist.max()])
    axes[1].set_ylim([0., fsdata.std(1)[:,param].max()])
    square(axes[1])
    axes[1].legend()
