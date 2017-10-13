"""
biastest.py

author: Colin Clement
date: 2017-06-26

This module measures bias in the statistical estimator
"""

import sys
sys.path.append("../")
import os

from datetime import datetime
import cPickle as pickle
from scipy.misc import face
import numpy as np
import tester

rng = np.random.RandomState(148509289)
                                                                      
def loadresults(directory):
    files = os.listdir(directory)
    files.remove('summary.pdf')
    data = {}
    for i, f in enumerate(files):
        with open(os.path.join(directory, f), "r") as infile:
            res = pickle.load(infile)
            noises = pickle.load(infile)
            datakwargs = pickle.load(infile)
            mask_kwargs = pickle.load(infile)
            alldata = pickle.load(infile)
        data[f] = {'results': res, 'noises': noises,
                   'datakwargs': datakwargs, 'mask_kwargs': mask_kwargs,
                   'alldata': alldata}
    return data


if __name__=="__main__":
    from datetime import datetime
    import matplotlib as mpl
    #mpl.use('Agg')
    import matplotlib.pyplot as plt

    today = datetime.today().isoformat().split("T")[0]

    dataL = 128 
    noise = 0.05
    deg = 40
    abscissa = deltas = np.linspace(0., 2., 40)
    xlabel = "True shift $\Delta_y$"
    padding = 6
    N = 2000

    #directory = 'deltaresults/x-zero-s_{}-p_{}-N_{}-'.format(noise,padding,N)+today

    datakwargs = [(2, dataL, noise), {'coef': rng.randn(2*deg-1)}]
    mask_kwargs = {'sigmoid': {'o': padding, 'w': 0.5},
                   'constant': {'o': padding, 'l': dataL-2*padding},
                   'linear': {}}

    #try:
    #    os.mkdir(directory)
    #except OSError as oerr:
    #    pass

    fig, axes = plt.subplots(1, 3, sharex='all', figsize=(10,5)) 

    #for i, data in enumerate(datakwargs):
    for j, mask in enumerate(mask_kwargs):
        start = datetime.now()
        biastest = tester.BiasTest(datakwargs, N=N, masktype=mask,
                               mask_kwargs=mask_kwargs[mask])
        results, alldata = biastest.deltaloop(deltas, noise)
        f, a = biastest.plotbias(results, abscissa=abscissa, xlabel=xlabel, 
                                 axis=axes[j], 
                                 title="{} mask".format(mask))
        plt.suptitle("{} padding, N={}, $\sigma$={}".format(padding, N, noise))
    #f.savefig(os.path.join(directory,"summary.pdf".format(data,mask)))
    #filename = os.path.join(directory,"data-{}_mask-{}.pkl".format(data,mask))
    #with open(filename, 'w') as outfile:
    #    pickle.dump(results, outfile)
    #    pickle.dump(deltas, outfile)
    #    pickle.dump(datakwargs[data], outfile)
    #    pickle.dump(mask_kwargs[mask], outfile)
    #    pickle.dump(alldata, outfile)
        print("finished {} in {}".format(mask, datetime.now()-start))
    plt.show()
