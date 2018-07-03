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

import super_reg.util.tester as tester
import super_reg.util.makedata as md

from super_reg.twod.fouriershift import Register
import super_reg.twod.chebseriespriors as csp
import super_reg.twod.chebseries as cs

rng = np.random.RandomState(148509289)

def checkdir(dirs):
    try:
        os.makedirs(dirs)
    except OSError as e:
        pass

resultsdir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'results')
checkdir(resultsdir)

def saveresults(filename, results, noises, datakwargs, mask_kwargs,
                alldata, shifts):
    with open(filename, "wb") as fout:
        pickle.dump(results, fout)
        pickle.dump(noises, fout)
        pickle.dump(datakwargs, fout)
        pickle.dump(mask_kwargs, fout)
        pickle.dump(alldata, fout)
        pickle.dump(shifts, fout)

def loadresults(directory):
    data = {}
    filenames = glob.glob(os.path.join(directory, '*.pkl'))
    for i, f in enumerate(filenames):
        with open(f, "rb") as infile:
            res = pickle.load(infile)
            noises = pickle.load(infile)
            datakwargs = pickle.load(infile)
            mask_kwargs = pickle.load(infile)
            alldata = pickle.load(infile)
            shifts = pickle.load(infile)
        name = os.path.split(f)[-1]
        data[name] = {'results': res, 'noises': noises,
                      'datakwargs': datakwargs, 'mask_kwargs': mask_kwargs,
                      'alldata': alldata, 'shifts': shifts}
    return data


if __name__=="__main__":
    from datetime import datetime
    import matplotlib as mpl
    #mpl.use('Agg')
    import matplotlib.pyplot as plt

    today = datetime.today().isoformat().split("T")[0]
    
    abscissa = np.linspace(0., 2., 16)
    shift = [rng.randn(2)]
    shifts = np.array([[[s, 0.]] for s in abscissa])
    noise = 0.05
    noises = np.linspace(5E-3, 0.1, 20)
    xlabel = "True shift $\Delta_y$"
    N = 300
    L = 32
    padding = 3
    deg = 15

    datakwargs = {'L': L, 'offset': np.zeros(2), 'shifts': shift,
                  'img': md.powerlaw((2*L, 2*L), 1.8, scale=L/4., rng=rng),
                  'mirror': False}
    mask_kwargs = [{'o': padding, 'l': L-2*padding}, 
                   {'o': padding, 'l': L-2*padding}]

    complexities = np.load("results/complexity-2018-06-04-noise-degree-gamma.npy")
    #complexities2 = np.load("results/complexity-endshift-2018-06-05-noise-degree-gamma.npy")
    # noise degree gamma

    fstest = tester.BiasTest(datakwargs, Register, N=N, mask_kwargs=mask_kwargs,
                             masktype='constant')
    dcomp = complexities[np.argmin(np.abs(complexities[:,0]-noise))]
    srtest = tester.BiasTest(datakwargs, cs.SuperRegistration, N=N, deg=deg)
    #srtest = tester.BiasTest(datakwargs, csp.SRGaussianPriors, N=N, 
    #                         deg=deg, gamma=0.001)

    fsdeltaloop = fstest.deltaloop(shifts, noise)
    srdeltaloop = srtest.deltaloop(shifts, noise, delta=1e-8).squeeze()

    np.savez("results/shiftbiascomparison-d_{}-N_{}-{}".format(deg, N, today),
             fs = fsdeltaloop, sr = srdeltaloop, noise=noise, shifts=shifts,
             datakwargs=[datakwargs])
