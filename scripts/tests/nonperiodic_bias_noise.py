"""
nonperiodic_bias_noise.py

author: Colin Clement
date: 2018-06-04

This module calculates the error and bias, as a function of noise and true
shift, for general non-periodic image registration, for both the standard method
of shifting one image to match the other, and for Super Registration.
"""

import os
import glob

from datetime import datetime
import pickle
from scipy.misc import face
import numpy as np
from itertools import chain

import superreg.util.tester as tester
import superreg.util.makedata as md

from superreg.fouriershift import Register
import superreg.chebseries as cs

rng = np.random.RandomState(148509289)

def checkdir(dirs):
    try:
        os.makedirs(dirs)
    except OSError as e:
        pass

resultsdir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'results')
checkdir(resultsdir)


if __name__=="__main__":
    from datetime import datetime
    import matplotlib as mpl
    #mpl.use('Agg')
    import matplotlib.pyplot as plt

    today = datetime.today().isoformat().split("T")[0]
    
    abscissa = np.linspace(0., 2., 20)
    shift = [rng.randn(2)]
    shifts = np.array([[[s, 0.]] for s in abscissa])
    noise = 0.075
    noises = np.linspace(5E-3, 0.1, 20)
    xlabel = "True shift $\Delta_y$"
    N = 1500
    L = 32
    padding = 3
    deg = 12

    datakwargs = {'L': L, 'offset': np.zeros(2), 'shifts': shift,
                  'img': md.powerlaw((2*L, 2*L), 1.8, scale=L/4., rng=rng),
                  'mirror': False}
    mask_kwargs = [{'o': padding, 'l': L-2*padding}, 
                   {'o': padding, 'l': L-2*padding}]

    fstest = tester.BiasTest(datakwargs, Register, N=N, mask_kwargs=mask_kwargs,
                             masktype='constant', noises=noises)

    srtest = tester.BiasTest(datakwargs, cs.SuperRegistration, N=N, deg=deg,
                             noises=noises)

    start = datetime.now()
    fsnoiseloop = fstest.noiseloop()
    print("Finished FS noiseloop in {}".format(datetime.now()-start))

    start = datetime.now()
    fsdeltaloop = fstest.deltaloop(shifts, noise)
    print("Finished FS deltaloop in {}".format(datetime.now()-start))

    start = datetime.now()
    srnoiseloop = srtest.noiseloop(delta=1e-8).squeeze()
    print("Finished SR noiseloop in {}".format(datetime.now()-start))

    start = datetime.now()
    srdeltaloop = srtest.deltaloop(shifts, noise, delta=1e-8).squeeze()
    print("Finished SR deltaloop in {}".format(datetime.now()-start))

    np.savez("results/shiftnoisebiascomparison-d_{}-N_{}-{}".format(deg, N, today),
             fsdelta=fsdeltaloop, srdelta=srdeltaloop, noise=noise,
             noises=noises, shift=shift, shifts=shifts, fsnoise=fsnoiseloop,
             srnoise=srnoiseloop, datakwargs=[datakwargs])
