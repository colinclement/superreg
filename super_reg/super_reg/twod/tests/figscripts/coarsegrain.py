import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['legend.fontsize'] =10 
mpl.rcParams['xtick.labelsize'] =12 
mpl.rcParams['ytick.labelsize'] =12 
mpl.rcParams['lines.linewidth'] = 1.2
mpl.rcParams['lines.markersize'] = 4.

def crb(I, sigma):
    ly, lx = I.shape
    Ik = np.fft.fftn(I, norm='ortho')  # Unitary norm!
    ky = np.fft.fftfreq(ly, d=1./(2*np.pi))[:,None]
    kx = np.fft.fftfreq(lx, d=1./(2*np.pi))
    return sigma **2 / np.sum(ky**2 * Ik * Ik.conj()).real

def roughness(I):
    ly, lx = I.shape
    Ik = np.fft.fftn(I, norm='ortho')  # Unitary norm!
    ky = np.fft.fftfreq(ly, d=1./(2*np.pi))[:,None]
    kx = np.fft.fftfreq(lx, d=1./(2*np.pi))
    return np.sum(ky**2 * Ik * Ik.conj()).real, np.sum(kx**2 * Ik * Ik.conj()).real

def predictvar(I, sigma, N=None):
    s = crb(I, sigma)
    N = N or I.size
    return 2*s*(1+N*np.pi**2 * s/6)

def theory(D, sigma, N, a):
    s = sigma**2/D
    return 2*s*(1+N*np.pi**2 * s/(6*a**4))

def collapse(var, D, sigma, N, a):
    s = sigma**2/D
    return (var/(2*s) - 1) * 6 * a**4 / (N *np.pi**2 * s)

#dat = np.load('results/coarsen-N_500-L_1024-2018-08-06.npz')
#dat = np.load('results/coarsen-N_500-L_1024-2018-08-07.npz')
#dat = np.load('results/coarsen-N_1000-L_512-2018-08-08.npz')
dat = np.load('results/coarsen-N_1000-L_1024-2018-08-08.npz')
#This one has one extra coarse graining and shows bias
#dat = np.load('results/coarsen-N_400-L_1024-2018-08-09.npz')
data = dat['data']
img = dat['datakwargs'][0]['img']
shift = dat['shift'].squeeze()
noises = dat['noises']
coarsenings = dat['coarsenings']
N = dat['N']

nn = np.linspace(noises.min(), noises.max(), 100)
Dy, Dx = roughness(img)

if True:
    fig, axe = plt.subplots(figsize=(4.5, 2.8))
    colors = [{'color': c} for c in
              mpl.cm.copper(mpl.colors.Normalize()(np.log(coarsenings)))]
    lines = []
    for i, (a, k) in enumerate(zip(coarsenings, colors)):
        #pvar = [theory(Dy, n, img.size,a) for n in nn]
        perr = [np.sqrt(theory(Dy, n, img.size,a)) for n in nn]
        #l, = axe.plot(nn, pvar, alpha=0.7, **k)
        l, = axe.loglog(nn, perr, alpha=0.7, **k)
        lines.append(l)
        #axe.plot(noises, data.std(1)[:,i,0,0]**2*a**2, 'o',
        #         label=r'$a={}$'.format(a), **k)
        axe.loglog(noises, data.std(1)[:,i,0,0]*a, 'o',
                 label=r'$a={}$'.format(a), **k)
    legend1 = axe.legend(loc='upper left', title="Measured")
    axe.add_artist(legend1)
    #l, = axe.plot(nn, 2*nn**2/Dy, '--', c='k', alpha=0.8, label=r"$2\sigma^2/D_y^2$")
    l, = axe.loglog(nn, np.sqrt(2)*nn/np.sqrt(Dy), '--', c='k', alpha=0.8, label=r"$2\sigma^2/D_y^2$")
    #legend2 = axe.legend((l,), (r"$2\frac{\sigma^2}{D_y^2}$",), loc='lower right')
    legend2 = axe.legend((l,), (r"$\sqrt{2}\frac{\sigma}{D_y}$",), loc='lower right')
    axe.add_artist(legend2)
    legend3 = axe.legend(lines, ('' for a in coarsenings), loc='upper left', 
                         title='Theory', bbox_to_anchor=(.3, 1.0))
    #legend3 = axe.legend(lines, ('$a={}$'.format(a) for a in coarsenings), loc='center left',
    #                     bbox_to_anchor=(1., .5),
    #                     title=r"$2\frac{\sigma^2}{D_y^2}\left(1+\frac{N \pi}{6 a^4}\frac{\sigma^2}{D_y^2}\right)$")

    axe.text(-.23, 1., "(b)", fontsize=14, transform=axe.transAxes)

    #axe.set_yscale('log')
    #axe.set_xscale('log')
    axe.set_xlabel("Noise $\sigma$")
    #axe.set_ylabel("Variance of $\Delta_y$")
    axe.set_ylabel("$\Delta_y$ Error")
    #axe.set_title('Coarsening can Improve Standard Method')
    ticks = np.logspace(np.log10(noises.min()), np.log10(noises.max()), 5)
    axe.set_xticks(ticks)
    axe.set_xticklabels(['{:.3f}'.format(t) for t in ticks]) 
    axe.set_xlim([noises.min(), noises.max()])

    plt.tight_layout()

if True:
    clist = [1, 4, 16]
    fig, axes = plt.subplots(1, len(clist))
    fakedata = img + np.random.randn(*img.shape)*0.05
    fakeimgs = [block_reduce(fakedata, (a,a)) for a in clist]
    #fmin, fmax = min(map(np.min, fakeimgs)), max(map(np.max, fakeimgs))
    fmin, fmax = None, None
    Ly, Lx = img.shape
    for i, (a, ax) in enumerate(zip(clist, axes)):
        sl = np.s_[-(Ly//a//8):,-(Lx//a//8):]
        ax.matshow(block_reduce(fakedata, (a,a))[sl], vmin=fmin, vmax=fmax,
                   cmap='bone')
        ax.axis('off')
        ax.set_title("$a={}$".format(a), fontsize=14,
                     verticalalignment='center')
    axes[0].text(-Lx//48, Ly//16-Ly//48, "$\sigma=0.05$", fontsize=14,
                 rotation='vertical')

    axes[0].text(-Lx//38, -Lx/90., "(a)", fontsize=16)

if False:
    fig, axe = plt.subplots(2, 2, figsize=(14, 10))
    for i, a in enumerate(coarsenings):
        axe[0,0].errorbar(noises, data.mean(1)[:,i,0,0]*a - shift[0], 
                        yerr=data.std(1)[:,i,0,0]*a/np.sqrt(N), 
                        label=str(a))
    axe[0,0].set_title("Bias after coarsening by $a$")
    axe[0,0].set_xlabel("Noise $\sigma$")
    axe[0,0].set_xlabel("Noise $\sigma$")
    axe[0,0].set_ylabel("Bias of $\Delta_y$")
    axe[0,0].legend(loc='upper left')
    
    for i, (a, k) in enumerate(zip(coarsenings, mpl.rcParams['axes.prop_cycle'])):
        axe[0,1].plot(noises, data.std(1)[:,i,0,0]**2*a**2, 'o', label=str(a), **k)
        #pvar = [theory(Dy, n/a, (img.shape[0]/a)**2)*a**2 for n in
        #        nn]
        pvar = [theory(Dy, n, img.size,a) for n in nn]
        axe[0,1].plot(nn, pvar, ':', label="Theory $a$={}".format(a), **k)
    axe[0,1].plot(nn, 2*nn**2/Dy, c='k', alpha=0.5, label=r"2$\times$CRB")
    axe[0,1].set_yscale('log')
    axe[0,1].set_xlabel("Noise $\sigma$")
    axe[0,1].set_ylabel("Variance of $\Delta_y$")
    axe[0,1].set_title('Variance after coarsening by $a$')
    axe[0,1].legend(loc='center left', bbox_to_anchor=(1., 0.5))
    
    
    for i, a in enumerate(coarsenings):
        axe[1, 0].errorbar(noises, data.mean(1)[:,i,0,1]*a - shift[1], 
                        yerr=data.std(1)[:,i,0,1]*a/np.sqrt(N), 
                        label=str(a))
    axe[1,0].set_title("Bias after coarsening by $a$")
    axe[1,0].set_ylabel("Bias of $\Delta_x$")
    axe[1,0].set_xlabel("Noise $\sigma$")
    axe[1,0].legend()
    
    nn = np.linspace(noises.min(), noises.max(), 100)
    for i, (a, k) in enumerate(zip(coarsenings, mpl.rcParams['axes.prop_cycle'])):
        axe[1,1].plot(noises, data.std(1)[:,i,0,1]**2*a**2, 'o', label=str(a), **k)
        #pvar = [theory(Dx, n/a, (img.shape[0]/a)**2)*a**2 for n in
        #        nn]
        pvar = [theory(Dx, n, img.size,a) for n in nn]
        axe[1,1].plot(nn, pvar, ':', label="Theory $a$={}".format(a), **k)
    axe[1,1].plot(nn, 2*nn**2/Dx, c='k', alpha=0.5, label=r"2$\times$CRB")
    axe[1,1].set_yscale('log')
    axe[1,1].set_xlabel("Noise $\sigma$")
    axe[1,1].set_ylabel("Variance of $\Delta_x$")
    axe[1,1].set_title('Variance after coarsening by $a$')
    axe[1,1].legend(loc='center left', bbox_to_anchor=(1., 0.5))
    
    plt.suptitle(r"Theory $\sigma_\Delta^2 = 2 \frac{\sigma^2}{D^2}\left(1+\frac{N \pi^2}{6 a ^4} \frac{\sigma^2}{D^2}\right)$ for $D=\sum_k k^2 |I_k|^2$, coarsening factor $a$ and $N=1024^2$")

plt.show()
