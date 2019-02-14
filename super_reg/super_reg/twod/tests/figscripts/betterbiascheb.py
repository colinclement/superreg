import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from super_reg.twod.tests.integerbiastest_cheb import loadresults
from super_reg.util.tester import BiasTest
from super_reg.twod.chebseries import SuperRegistration

mpl.rcParams['font.size'] = 16.
mpl.rcParams['axes.labelsize'] = 20.
mpl.rcParams['axes.titlesize'] = 20.
mpl.rcParams['legend.fontsize'] = 16

data = loadresults('../cheb-fourier-shift-bias-2018-01-06')
cheb = data['cheb-super-N_250-L_32-n_0.08.pkl']
fourier = data['fourier-shift-N_250-L_32-n_0.08.pkl']

bt = BiasTest(cheb['datakwargs'], SuperRegistration, deg=13)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
xlabel = "True Shift $\Delta_y$"
shifts = cheb['shifts'].squeeze()[:,0]
sl = cheb['datakwargs']['sliceobj']


axes[0].matshow(cheb['datakwargs']['img'][sl], cmap='Greys')
axes[0].axis('off')
axes[0].set_title("Latent Image with $\sigma=${:.3f}".format(cheb['noises'][0]))

bt.plotbias(fourier['results'], abscissa=shifts, xlabel=xlabel,
            axis=axes[1], title="Fourier Shift")
bt.plotbias(cheb['results'], abscissa=shifts, xlabel=xlabel,
            axis=axes[2], title="Super Registration")
axes[2].set_ylim(axes[1].get_ylim())

l = axes[1].legend()
l.remove()
axes[2].legend(loc='upper left')

for ax in axes[1:]:
    ax.axhline(0., c='k', lw=0.8)

plt.tight_layout()

plt.show()
