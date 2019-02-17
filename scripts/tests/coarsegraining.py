"""
coarsegraining.py

author: Colin Clement
date: 2018-08-05

This module calculates the precision of the fourier shift method as you coarse
grain your data
"""

import os

from datetime import datetime
import numpy as np

from superreg.util.tester import BiasTest
import superreg.util.makedata as md

from superreg.periodicshift import Register

rng = np.random.RandomState(148509289)

def checkdir(dirs):
    try:
        os.makedirs(dirs)
    except OSError as e:
        pass

def crb(I, sigma):
    ly, lx = I.shape
    Ik = np.fft.fftn(I, norm='ortho')  # Unitary norm!
    ky = np.fft.fftfreq(ly, d=1./(2*np.pi))[:,None]
    kx = np.fft.fftfreq(lx, d=1./(2*np.pi))
    return sigma **2 / np.sum(ky**2 * Ik * Ik.conj()).real

def predictvar(I, sigma, N=None):
    s = crb(I, sigma)
    N = N or I.size
    return 2*s*(1+N*np.pi**2 * s/6)

resultsdir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'results')
checkdir(resultsdir)

if __name__=="__main__":
    from datetime import datetime
    import matplotlib as mpl
    #mpl.use('Agg')
    import matplotlib.pyplot as plt

    today = datetime.today().isoformat().split("T")[0]
    shift = [rng.randn(2)]
    noises = np.linspace(5E-3, 0.1, 20)
    N = 400
    L = 1024
    #coarsenings = range(1,9)  # non-powers of 2 lead to very inaccurate results
    coarsenings = [1, 2, 4, 8, 16, 32]

    saveloc = 'results/coarsen-N_{}-L_{}-'.format(N, L)+today

    datakwargs = {
        'L': L, 'offset': np.zeros(2), 'shifts': shift, 
        'img': md.powerlaw((L, L), 1.8, scale=L/64., rng=rng), 'mirror': False
    }
    img = datakwargs['img']
    
    tester = BiasTest(datakwargs, N=N, registration=Register,
                      noises = noises, plan="FFTW_PATIENT")

    # first do a standard noise loop to debug the first step of coarse graining
    #noiseloop = tester.noiseloop()

    start = datetime.now()
    data = tester.coarsegrain(coarsenings)
    print("Finished testing coarse graining in {}".format(datetime.now()-start))

    np.savez(saveloc, datakwargs=[datakwargs], noises=noises, 
             shift=shift, data=data, coarsenings=list(coarsenings), N=N)
             #noiseloop=noiseloop)
