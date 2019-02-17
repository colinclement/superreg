import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from superreg.chebseries import SuperRegistration
from superreg.fouriershift import Register
import superreg.util.image as ui
import makedata as md

mpl.rcParams['image.cmap'] = 'Greys'

rng = np.random.RandomState(92089)

N = 30
L = 24
sigma = 0.1
s0 = np.concatenate([[np.zeros(2)], rng.randn(N-1, 2)/64.])
shifts = L * s0
model = md.twositesquarelattice(s0, L, noise=1./250)
data = model + sigma * rng.randn(*model.shape)
# randoom numbers different!
truth = md.twositesquarelattice([np.zeros(2)], 2*L, noise=1./250)[0]

fshift = ui.multi_image_pairwise_registration(data, s0list=shifts[1:])
fsrecon = ui.shift_all_to(data, fshift)

# 29 is the max evd for N=20, L=32, sigma=0.1
# 29 works the best but isn't the max evidence for N=30, L=18, sigma=0.1
if True:
    deglist = range(23, 31)
    
    evdlist = []
    shiftlist = []
    coeflist = []
    
    start = datetime.now()
    for i, d in enumerate(deglist):
        print("deg = {}".format(d))
        reg = SuperRegistration(data, d, shifts=fshift) #shifts[1:])
        shiftlist.append(reg.fit(iprint=1, delta=1E-5, lamb=0.1))
        evdlist.append(reg.evidence(sigma=sigma))
        coeflist.append(reg.coef)
        print("evd = {}".format(evdlist[-1]))
    print("Calculation took {}".format(datetime.now()-start))
    
    today = format(datetime.now()).split()[0]
    reg = SuperRegistration(data, list(deglist)[np.argmax(evdlist)], 
                            shifts=shiftlist[np.argmax(evdlist)][0],
                            coef=coeflist[np.argmax(evdlist)])
else:
    reg = SuperRegistration(data, 29, shifts=fshift)
    reg.fit(iprint=1, delta=1e-5, lamb=0.1)

