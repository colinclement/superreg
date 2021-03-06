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

from superreg.util.tester import BiasTest
import superreg.util.makedata as md
from superreg.fouriershift import Register

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
            img = pickle.load(img)
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
    noise_scale = .5
    N = 100

    directory = 'results/smoothed{:.2f}-N_{}-'.format(noise_scale,N)+today

    datakwargs = {
        'random': {'L': L, 'offset': np.zeros(2), 'shifts': [delta],
                   'img': md.powerlaw((L, L), 1.8, scale=L/4., rng=rng),
                   'mirror': False,
                   #'noisegen': lambda x, y, z: 2*np.random.rand(x, y, z)-1, }
                   'noisegen': md.correlatednoise,
                   'noiseargs': (noise_scale,)}
        }
    mask_kwargs = {'none': [{}, {}]}
    
    os.makedirs(directory, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(24,8)) 
    mask = 'none'
    data = 'random'

    start = datetime.now()
    biastest = BiasTest(datakwargs[data], N=N, masktype=mask,
                        mask_kwargs=mask_kwargs[mask],
                        registration=Register,
                        noises = noises, mirror=False)
    alldata = biastest.noiseloop(delta0=delta+0.01*np.random.randn(2),)
    print("Finished noise loop")
    alldata_delta = biastest.deltaloop(shifts, noises[6]).squeeze()
    print("Finished shift loop")
    
    results_y = {'bias': [], 'bias_std': [], 'biaserr': [], 'err': []}
    results_x = {'bias': [], 'bias_std': [], 'biaserr': [], 'err': []}
    for dd, dds in zip(alldata, alldata_delta):
        p1s, p1_sigmas = dd
        results_y['bias'] += [np.mean(p1s[:,0])-delta[0]]
        results_y['bias_std'] += [np.std(p1s[:,0])]
        results_y['biaserr'] += [np.std(p1s[:,0])/np.sqrt(len(p1s))]
        results_y['err'] += [np.mean(p1_sigmas[:,0])]
        results_x['bias'] += [np.mean(p1s[:,1])-delta[1]]
        results_x['bias_std'] += [np.std(p1s[:,1])]
        results_x['biaserr'] += [np.std(p1s[:,1])/np.sqrt(len(p1s))]
        results_x['err'] += [np.mean(p1_sigmas[:,1])]
        
    results_deltay = {'bias': [], 'bias_std': [], 'biaserr': [], 'err': []}
    for dds, d in zip(alldata_delta, shifts.squeeze()):
        p1s, p1_sigmas = dds
        results_deltay['bias'] += [np.mean(p1s[:,1])-d[1]]
        results_deltay['bias_std'] += [np.std(p1s[:,1])]
        results_deltay['biaserr'] += [np.std(p1s[:,1])/np.sqrt(len(p1s))]
        results_deltay['err'] += [np.mean(p1_sigmas[:,1])]

    img = datakwargs[data]['img']
    axes[0].matshow(img, cmap='Greys')
    axes[0].axis('off')
    f, a = biastest.plotbias(results_x, axis=axes[1],
                           title="{} data with {} mask".format(data,mask))
    f, a = biastest.plotbias(results_deltay, abscissa=abscissa, xlabel=xlabel, 
                             axis=axes[2], title="Shift Dependent Bias")

    f.savefig(os.path.join(directory,"summary.pdf".format(data,mask)))
    filename = os.path.join(directory,"data-{}_mask-{}.pkl".format(data,mask))
    filename_deltas = os.path.join(directory,
                                   "shift-data-{}_mask-{}.pkl".format(data,mask))
    saveresults(filename, results_x, noises, datakwargs, mask_kwargs,
                alldata, shifts, img)
    saveresults(filename_deltas, results_deltay, noises, datakwargs, mask_kwargs,
                alldata_delta, shifts, img)
    #with open(filename, 'wb') as outfile:
    #    pickle.dump(results_y, outfile)
    #    pickle.dump(biastest.noises, outfile)
    #    pickle.dump(datakwargs[data], outfile)
    #    pickle.dump(mask_kwargs[mask], outfile)
    #    pickle.dump(alldata, outfile)
    print("finished {}, {} in {}".format(data, mask, datetime.now()-start))
