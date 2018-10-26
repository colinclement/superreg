import os
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 10 
mpl.rcParams['xtick.labelsize'] =12 
mpl.rcParams['ytick.labelsize'] =12 
mpl.rcParams['lines.linewidth'] = 1.2
mpl.rcParams['lines.markersize'] = 4.

def predictyerr(I, sigma):
    ly, lx = I.shape
    n = ly*lx
    dy = np.gradient(I, axis=0)
    denom = np.sum(dy*dy)
    sumks = n * np.pi**2/3.
    return np.sqrt(2*sigma**2/denom + sigma**4 * sumks / denom**2)

# noise = 0.05
#filename = 'shiftnoisebiascomparison-d_10-N_1000-2018-07-09.npz'
#filename = 'shiftnoisebiascomparison-d_11-N_1000-2018-07-11.npz'
#Best one so far:
#filename = 'shiftnoisebiascomparison-d_12-N_1000-2018-07-09.npz'
#filename = 'shiftnoisebiascomparison-d_13-N_1000-2018-07-10.npz'

# noise = 0.075
#filename = 'shiftnoisebiascomparison-d_9-N_1000-2018-07-12.npz'
#filename = 'shiftnoisebiascomparison-d_10-N_1000-2018-07-13.npz'
#filename = 'shiftnoisebiascomparison-d_11-N_1000-2018-07-13.npz'
#filename = 'shiftnoisebiascomparison-d_12-N_1000-2018-07-14.npz'
#Best of this noise level
filename = 'shiftnoisebiascomparison-d_12-N_1500-2018-07-16.npz'

dat = np.load(os.path.join('results', filename))
img = dat['datakwargs'][0]['img']

shift = dat['shift'].squeeze()[0]
shifts = dat['shifts'].squeeze()[:,0]
noise = dat['noise']
noises = dat['noises']

fsnoise = dat['fsnoise']
fsdelta = dat['fsdelta']
srnoise = dat['srnoise']
srdelta = dat['srdelta']

margcolor = "#f3899a"
srcolor = "#007f6f"
flw = 0.

fig, axes = plt.subplots(2,1, figsize=(4., 8), ) # sharey=True)

srb = axes[0].errorbar(shifts, srdelta.mean(2)[:,0,0]-shifts, c=srcolor,
                       yerr=srdelta.std(2)[:,0,0]/np.sqrt(srdelta.shape[2]),)
sre = axes[0].scatter(shifts, srdelta.std(2)[:,0,0], color=srcolor, marker='+',
                      s=30.)
srcrb = srdelta.mean(2)[:,1,0]
src = axes[0].fill_between(shifts, srcrb, y2=-srcrb, color=srcolor, alpha=0.2,
                           lw=flw)

fsb = axes[0].errorbar(shifts, fsdelta.mean(2)[:,0,0]-shifts, c=margcolor,
                       yerr=fsdelta.std(2)[:,0,0]/np.sqrt(fsdelta.shape[2]),)
fse = axes[0].scatter(shifts, fsdelta.std(2)[:,0,0], marker='o', c=margcolor,)
fscrb = fsdelta.mean(2)[:,1,0]
fsc = axes[0].fill_between(shifts, fscrb, y2=-fscrb, color=margcolor,
                           alpha=0.3, lw=flw)

axes[0].text(-.4, .48, "(a)", fontsize=14)

#srlegend = axes[0].legend(
#    (srb[0], sre, src), ('Super registration (SR) bias', 'SR error', 'SR CRB'), 
#    loc='upper left', bbox_to_anchor=(-.02, 1.02))
#axes[0].add_artist(srlegend)
#fslegend = axes[0].legend(
#    (fsb[0], fse, fsc), ('Fourier shift bias', 'FS error', 'FS CRB'), 
#    loc='upper left', bbox_to_anchor=(0.15, 1.))

axes[0].set_xlim([shifts.min(), shifts.max()])
axes[0].set_ylabel("$\Delta_y$ error and bias ")
axes[0].set_xlabel("$y$-direction True shift $\Delta_0$ ")
axes[0].set_title("Shift-dependence for $\sigma$ = {:.3f}".format(noise))

srb = axes[1].errorbar(noises, srnoise.mean(2)[:,0,0]-shift, c=srcolor,
                       yerr=srnoise.std(2)[:,0,0]/np.sqrt(srnoise.shape[2]),)
sre = axes[1].scatter(noises, srnoise.std(2)[:,0,0], color=srcolor, marker='+',
                      s=30.)
srcrb = srnoise.mean(2)[:,1,0]
src = axes[1].fill_between(noises, srcrb, y2=-srcrb, color=srcolor, alpha=0.2,
                           lw=flw)
              

fstheory = axes[1].plot(noises, [predictyerr(img[:32,:32], n) for n in noises],
                        c='k', linestyle=':')
fsb = axes[1].errorbar(noises, fsnoise.mean(2)[:,0,0]-shift, c=margcolor,
                       yerr=fsnoise.std(2)[:,0,0]/np.sqrt(fsnoise.shape[2]),)
fse = axes[1].scatter(noises, fsnoise.std(2)[:,0,0], marker='o', c=margcolor,)
fscrb = fsnoise.mean(2)[:,1,0]
fsc = axes[1].fill_between(noises, fscrb, y2=-fscrb, color=margcolor,
                           alpha=0.3, lw=flw)
axes[1].text(-.01, .48, "(b)", fontsize=14)
              

axes[1].set_xlim([0., noises.max()])
axes[1].set_xlabel("Noise $\sigma$")
#axes[1].set_ylabel("Inferred $\Delta$ error and bias")
axes[1].set_title(
    "$\Delta_0$=({:.2f},{:.2f})".format(*dat['shift'][0])
)

#srlegend = axes[1].legend(
#    (srb[0], sre, src), ('Super registration bias', 'SR error', 'SR CRB'), 
#    bbox_to_anchor=(0.05, .8), loc='upper left'
#) # #axes[1].add_artist(srlegend)
#fslegend = axes[1].legend(
#    (fsb[0], fse, fstheory[0], fsc), ('Fourier shift bias', 'FS error',
#                                   'Theoretical FS error', 'FS CRB'), 
#    bbox_to_anchor=(0.0, 1.02), loc='upper left'
#)

axes[1].legend(
    (srb[0], sre, src, fsb[0], fse, fsc, fstheory[0]),
    ('Super registration\n(SR) bias', 'SR error', 'SR CRB', 'Fourier shift bias', 
     'FS error', 'FS CRB', 'FS theory error'), rcol=2,
    loc='lower center', bbox_to_anchor=(.5, -1.)
)

plt.tight_layout()

today = datetime.datetime.now().isoformat().split('T')[0]
dirname = "/home/colin/work/overleaf-papers/superreg/figs"
plt.savefig(os.path.join(dirname,
                         "{}-nonperiodic-bias-error.pdf".format(today)),
           bbox_inches='tight')
plt.show()
