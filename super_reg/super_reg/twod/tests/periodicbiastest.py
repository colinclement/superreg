"""
biastest.py

author: Colin Clement
date: 2017-06-26

This module measures bias in the statistical estimator
"""

import pickle
import sys
import os

import numpy as np
from datetime import datetime

from super_reg.util.tester import BiasTest
import super_reg.util.makedata as md
from super_reg.twod.periodicshift import Register
from super_reg.twod.fourierseries import SuperRegistration

rng = np.random.RandomState(148509289)
                                                                      
def saveresults(filename, results, noises, datakwargs, mask_kwargs,
                alldata, shifts, img):
    with open(filename, "wb") as fout:
        pickle.dump(results, fout)
        pickle.dump(noises, fout)
        pickle.dump(datakwargs, fout)
        pickle.dump(mask_kwargs, fout)
        pickle.dump(alldata, fout)
        pickle.dump(shifts, fout)
        pickle.dump(img, fout)

def loadresults(directory):
    files = os.listdir(directory)
    files.remove('summary.pdf')
    data = {}
    for i, f in enumerate(files):
        with open(os.path.join(directory, f), "rb") as infile:
            res = pickle.load(infile)
            noises = pickle.load(infile)
            datakwargs = pickle.load(infile)
            mask_kwargs = pickle.load(infile)
            alldata = pickle.load(infile)
            shifts = pickle.load(infile)
            img = pickle.load(infile)
        data[f] = {'results': res, 'noises': noises,
                   'datakwargs': datakwargs, 'mask_kwargs': mask_kwargs,
                   'alldata': alldata, 'shifts': shifts, 'img': img}
    return data


if __name__=="__main__":
    from datetime import datetime
    import matplotlib as mpl
    #mpl.use('Agg')
    import matplotlib.pyplot as plt

    today = datetime.today().isoformat().split("T")[0]

    L = 128
    delta = rng.rand(2)*2
    padding = 0
    abscissa = np.linspace(0., 2., 50)
    xlabel = "True shift $\Delta_y$"
    shifts = np.array([[[delta[0], s]] for s in abscissa])
    noises = np.linspace(0, 0.1, 20)
    N = 1000
    deg = 13
    # For this data set 13 maximized the evidence at sigma=.1

    directory = 'results/N_{}-deg_{}-'.format(N, deg)+today

    datakwargs = {
        'random': {'L': L, 'offset': np.zeros(2), 'shifts': [delta],
                   'img': md.powerlaw((L, L), 1.8, scale=L/6., rng=rng),
                   'mirror': False}
        }
    mask_kwargs = {'none': [{}, {}]}
    
    os.makedirs(directory, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(24,8)) 
    mask = 'none'
    data = 'random'

    start = datetime.now()
    biastest = BiasTest(datakwargs[data], N=N, registration=Register,
                        noises = noises)
    alldata = biastest.noiseloop(delta0=delta+0.01*np.random.randn(2),)
    print("Finished noise loop in {}".format(datetime.now()-start))
    
    start = datetime.now()
    biastest_sr = BiasTest(datakwargs[data], N=N,
                           registration=SuperRegistration,
                           noises=noises, deg=deg)
    # Note deg=17 was tested by maximizing evidence in fourierseries.py
    p0 = biastest_sr.reg.p0.copy()
    p0 /= np.sqrt(len(p0))
    alldata_sr = biastest_sr.noiseloop(p0=p0, sigma=0.).squeeze()
    print("Finished noise loop in {}".format(datetime.now()-start))

    results_y = {'bias': [], 'bias_std': [], 'biaserr': [], 'err': []}
    results_x = {'bias': [], 'bias_std': [], 'biaserr': [], 'err': []}
    for dd in alldata:
        p1s, p1_sigmas = dd
        results_y['bias'] += [np.mean(p1s[:,0])-delta[0]]
        results_y['bias_std'] += [np.std(p1s[:,0])]
        results_y['biaserr'] += [np.std(p1s[:,0])/np.sqrt(len(p1s))]
        results_y['err'] += [np.mean(p1_sigmas[:,0])]
        results_x['bias'] += [np.mean(p1s[:,1])-delta[1]]
        results_x['bias_std'] += [np.std(p1s[:,1])]
        results_x['biaserr'] += [np.std(p1s[:,1])/np.sqrt(len(p1s))]
        results_x['err'] += [np.mean(p1_sigmas[:,1])]
        
    results_superreg = {'bias': [], 'bias_std': [], 'biaserr': [], 'err': []}
    for dds in alldata_sr:
        p1s, p1_sigmas = dds
        results_superreg['bias'] += [-np.mean(p1s[:,1])-delta[1]]
        # Convention in superreg changes sign of answer
        results_superreg['bias_std'] += [np.std(p1s[:,1])]
        results_superreg['biaserr'] += [np.std(p1s[:,1])/np.sqrt(len(p1s))]
        results_superreg['err'] += [np.mean(p1_sigmas[:,1])]

    img = datakwargs[data]['img']
    axes[0].matshow(img, cmap='Greys')
    axes[0].axis('off')
    f, a = biastest.plotbias(results_x, axis=axes[1],
                             title="{} data with {} mask".format(data,mask))
    f, a = biastest.plotbias(results_superreg,
                             axis=axes[2], title="Super Registration")

    f.savefig(os.path.join(directory,"summary.pdf".format(data,mask)))
    filename = os.path.join( directory,"periodic-shift.pkl")
    filename_superreg = os.path.join( directory, "periodic-superreg.pkl")
    saveresults(filename, results_x, noises, datakwargs, mask_kwargs,
                alldata, shifts, img)
    saveresults(filename_superreg, results_superreg, noises, datakwargs, 
                mask_kwargs, alldata_sr, shifts, img)
