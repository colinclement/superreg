import numpy as np

import super_reg.twod.fourierseries as fs
from super_reg.twod.chebseries import SuperRegistration

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import copper

mpl.rcParams['font.size'] = 16.
mpl.rcParams['axes.labelsize'] = 20.
mpl.rcParams['axes.titlesize'] = 20.
mpl.rcParams['legend.fontsize'] = 16

deg = 8
L = 32
datamaker0 = fs.SuperRegistration(np.zeros((2, L, L)), deg=deg)
datamaker0.shifts = np.array([3*np.random.randn(2)])
shifts = datamaker0.shifts
fdata = datamaker0.model
fdata /= fdata.std()

data = fdata + 0.05 * np.random.randn(*fdata.shape)

reg = SuperRegistration(data, 16)
#print(reg.fit(iprint=2, delta=1E-4))
evd, bias = [], []
fig, ax = plt.subplots(1, 2, sharex=True)
order = list(range(10, 25))
nlist = list(range(2,9,2))  # [2]  
colors = copper(Normalize()(nlist))
shifts = np.random.randn(max(nlist)-1, 2)

datalist = []
biases = []
evidences = []

# This sequence plots the evidence as a function of complexity as a
# function of the number of images

for n, c in zip(nlist, colors):
    dm = fs.SuperRegistration(np.zeros((n, L, L)), deg)
    dm.coef = datamaker0.coef
    dm.shifts = shifts[:n-1]
    data = dm.model/dm.model.std()
    data += 0.05 * np.random.randn(*data.shape)
    datalist += [data]

    bias, evd = [], []
    print('{} images'.format(n))
    for i in order:
        reg = SuperRegistration(data, i)
        s, _ = reg.fit(delta=1E-7)
        e = reg.evidence(sigma=0.05)
        print('\ti = {}, e = {}'.format(i, e))
        bias.append(np.abs((s - dm.shifts)[0]))
        evd.append(e)
    biases.append(bias)
    evidences.append(evd)

    ax[0].plot(order, np.array(bias)[:,0], c = c, label='n = {}'.format(n))
    ax[0].plot(order, np.array(bias)[:,1], c = c)
    ax[1].plot(order, np.array(evd)/n, c = c)  # , label='N = {}'.format(n))

ax[0].legend(title="Number of images", loc='upper right')
ax[0].set_xlabel("Maximum Order")
ax[1].set_xlabel("Maximum Order")
ax[0].set_title('Error in shift reconstruction')
ax[1].set_title('Evidence per image')
ax[1].set_ylim([-2000,0])
plt.show()

#  Other poster complexity curves
#plt.figure()
#plt.plot(order, np.array(evidences[0]), lw=2.)
#plt.xlim([11,24])
#plt.ylim([-5000,-2000])
#plt.xlabel("Maximum Order of Chebyshev Polynomial")
#plt.ylabel(r"$\log p(\phi,\psi)$")
#
#fig, axes = plt.subplots(1, 3)
#for a, d in zip(axes, [11, 14, 24]):
#    reg = SuperRegistration(datalist[0], d)
#    reg.fit(delta=1E-7)
#    a.matshow(reg.residual[0])
#    a.axis('off')
#    a.set_title("Residual for N = {}".format(d))
#plt.show()
