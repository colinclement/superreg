import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 9 

#filename = 'shiftnoisebiascomparison-d_10-N_1000-2018-07-09.npz'
filename = 'shiftnoisebiascomparison-d_12-N_1000-2018-07-09.npz'
#filename = 'shiftnoisebiascomparison-d_13-N_1000-2018-07-10.npz'
dat = np.load(os.path.join('results', filename))

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

fig, axes = plt.subplots(1,2, figsize=(7.25, 2.8))

srb = axes[0].errorbar(shifts, srdelta.mean(2)[:,0,0]-shifts, c=srcolor,
                       yerr=srdelta.std(2)[:,0,0]/np.sqrt(srdelta.shape[2]),)
sre = axes[0].scatter(shifts, srdelta.std(2)[:,0,0], color=srcolor, marker='+')
#src, = axes[0].plot(shifts, srdelta.mean(2)[:,1,0], '-.', c=srcolor)
srcrb = srdelta.mean(2)[:,1,0]
src = axes[0].fill_between(shifts, srcrb, y2=-srcrb, color=srcolor, alpha=0.2,
                           lw=1.5)

fsb = axes[0].errorbar(shifts, fsdelta.mean(2)[:,0,0]-shifts, c=margcolor,
                       yerr=fsdelta.std(2)[:,0,0]/np.sqrt(fsdelta.shape[2]),)
fse = axes[0].scatter(shifts, fsdelta.std(2)[:,0,0], marker='+', c=margcolor,)
fscrb = fsdelta.mean(2)[:,1,0]
#fsc, = axes[0].plot(shifts, fsdelta.mean(2)[:,1,0], '-.', c=margcolor)
fsc = axes[0].fill_between(shifts, fscrb, y2=-fscrb, color=margcolor, alpha=0.3, lw=1.5)

#srlegend = axes[0].legend((srb[0], sre, src), ('Super registration bias', 'SR error',
#                                        'SR CRB'), loc='lower right')
#axes[0].add_artist(srlegend)
#fslegend = axes[0].legend((fsb[0], fse, fsc), ('Fourier shift bias', 'FS error',
#                                        'FS CRB'), loc='upper right')

axes[0].set_xlim([shifts.min(), shifts.max()])
axes[0].set_ylabel("Inferred $\Delta$ error and bias")
axes[0].set_xlabel("True shift $\Delta_0$ in $y$-direction")
axes[0].set_title("Shift-dependence for $\sigma$ = 0.5")

srb = axes[1].errorbar(noises, srnoise.mean(2)[:,0,0]-shift, c=srcolor,
                       yerr=srnoise.std(2)[:,0,0]/np.sqrt(srnoise.shape[2]),)
sre = axes[1].scatter(noises, srnoise.std(2)[:,0,0], color=srcolor, marker='+')
#src, = axes[1].plot(noises, srnoise.mean(2)[:,1,0], '-.', c=srcolor)
srcrb = srnoise.mean(2)[:,1,0]
src = axes[1].fill_between(noises, srcrb, y2=-srcrb, color=srcolor, alpha=0.2,
                           lw=1.5)

fsb = axes[1].errorbar(noises, fsnoise.mean(2)[:,0,0]-shift, c=margcolor,
                       yerr=fsnoise.std(2)[:,0,0]/np.sqrt(fsnoise.shape[2]),)
fse = axes[1].scatter(noises, fsnoise.std(2)[:,0,0], marker='+', c=margcolor,)
#fsc, = axes[1].plot(noises, fsnoise.mean(2)[:,1,0], '-.', c=margcolor)
fscrb = fsnoise.mean(2)[:,1,0]
fsc = axes[1].fill_between(noises, fscrb, y2=-fscrb, color=margcolor, alpha=0.3,
                           lw=1.5)

axes[1].set_xlim([0., noises.max()])
axes[1].set_xlabel("Noise $\sigma$")
axes[1].set_ylabel("Inferred $\Delta$ error and bias")
axes[1].set_title(
    "$\Delta_0$=({:.2f},{:.2f})".format(*dat['shift'][0])
)

#srlegend = axes[1].legend(
#    (srb[0], sre, src), ('Super registration bias', 'SR error', 'SR CRB'), 
#    bbox_to_anchor=(0.05, .8), loc='upper left'
#) # #axes[1].add_artist(srlegend)
#fslegend = axes[1].legend(
#    (fsb[0], fse, fsc), ('Fourier shift bias', 'FS error', 'FS CRB'), 
#    bbox_to_anchor=(0.05, .8), loc='lower left'
#)

axes[1].legend(
    (srb[0], sre, src, fsb[0], fse, fsc),
    ('Super registration bias', 'SR error', 'SR CRB', 'Fourier shift bias', 
     'FS error', 'FS CRB'), loc='upper left', bbox_to_anchor=(-.02, 1.04)
)

plt.tight_layout()
plt.show()
