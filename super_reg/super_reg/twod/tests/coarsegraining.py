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

from super_reg.util.tester import BiasTest
import super_reg.util.makedata as md

from super_reg.twod.periodicshift import Register

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
    shift = [rng.randn(2)]
    noises = np.linspace(5E-3, 0.1, 20)
    N = 200
    L = 1024
    coarsenings = range(1,9)

    directory = 'results/coarsen-N_{}-L_{}-'.format(N, L)+today

    datakwargs = {
        'L': L, 'offset': np.zeros(2), 'shifts': shift, 
        'img': md.powerlaw((L, L), 1.8, scale=L/6., rng=rng), 'mirror': False
    }
    
    os.makedirs(directory, exist_ok=True)

    start = datetime.now()
    tester = BiasTest(datakwargs, N=N, registration=Register,
                      noises = noises)

    data = tester.coarsegrain(coarsenings)
