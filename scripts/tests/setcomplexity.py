"""
setcomplexity.py

author: Colin Clement
date: 2018-06-04

This module predicts the correct complexity to use in subsequent noise/bias
studies
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

def find_all_complexities(datakwargs, noises
    complexities = []
    for n in noises:
        data = md.fakedata(n, **datakwargs)
        d, g = cs.optcomplexity(data, cs.SRGaussianPriors, sigma=n,
                                show=True, delta=1E-6, tol=1E-4)
        print(n, d, g)
        complexities.append([n, d, g])
    complexities = np.array(complexities)


if __name__=="__main__":
    from datetime import datetime
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    today = datetime.today().isoformat().split("T")[0]
    
    abscissa = np.linspace(0., 2., 16)
    shift = [rng.randn(2)]
    shifts = np.array([[[s, 0.]] for s in abscissa])
    noise = 0.08
    noises = np.linspace(5E-3, 0.1, 20)
    xlabel = "True shift $\Delta_y$"
    N = 250
    L = 32
    padding = 3

    datakwargs = {'L': L, 'offset': np.zeros(2), 'shifts': shifts[-1],
                  'img': md.powerlaw((2*L, 2*L), 1.8, scale=2*L/6., rng=rng),
                  'mirror': False}

    complexities = []
    for n in noises:
        data = md.fakedata(n, **datakwargs)
        d, g = cs.optcomplexity(data, cs.SRGaussianPriors, sigma=n,
                                show=True, delta=1E-6, tol=1E-4)
        print(n, d, g)
        complexities.append([n, d, g])
    complexities = np.array(complexities)

    savename = "results/complexity-{}-noise-degree.npy".format(today)
    np.save(savename, complexities)
