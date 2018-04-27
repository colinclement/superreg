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
    import matplotlib.pyplot as plt

    today = datetime.today().isoformat().split("T")[0]

    L = 128
    delta = rng.rand(2)*2
    padding = 0
    abscissa = np.linspace(0., 2., 50)
    deglist = np.array(range(2, 64))
    xlabel = "True shift $\Delta_y$"
    shifts = np.array([[[delta[0], s]] for s in abscissa])
    noises = np.linspace(0, 0.1, 20)
    noise = 0.075
    N = 2000
    deg = 13

    directory = 'results/complexity-N_{}-deglist_{}-noise_{}-'.format(N, len(deglist),
                                                           noise)+today

    datakwargs = {'L': L, 'offset': np.zeros(2), 'shifts': [delta],
                   'img': md.powerlaw((L, L), 1.8, scale=L/6., rng=rng),
                   'mirror': False}
    mask_kwargs = {'none': [{}, {}]}
    
    os.makedirs(directory, exist_ok=True)

    start = datetime.now()
    biastest = BiasTest(datakwargs, N=N,
                        registration=SuperRegistration,
                        noises=noises, deg=deg)
    p0 = biastest.reg.p0.copy()
    p0 /= np.sqrt(len(p0))

    kwarglist = [dict(deg=d) for d in deglist]
    alldata = biastest.kwargloop(kwarglist, noise=noise)
    print("Finished noise loop in {}".format(datetime.now()-start))

    results = {'bias': [], 'bias_std': [], 'biaserr': [], 'err': [],
               'evidence': [], 'evidence_std': []}
    for dds in alldata:
        p1s, p1_sigmas, evds = np.array(dds[0]), np.array(dds[1]), np.array(dds[2])
        results['bias'] += [-np.mean(p1s[:,1])-delta[1]]
        results['evidence'] += [np.mean(evds)]
        results['evidence_std'] += [np.std(evds)]
        # Convention in superreg changes sign of answer
        results['bias_std'] += [np.std(p1s[:,1])]
        results['biaserr'] += [np.std(p1s[:,1])/np.sqrt(len(p1s))]
        results['err'] += [np.mean(p1_sigmas[:,1])]

    img = datakwargs['img']

    fig, axe = plt.subplots()
    axe.errorbar(deglist, results['evidence'], yerr=results['evidence_std'])
    axe2 = axe.twinx()
    axe2.plot(deglist, results['bias_std'])
    axe2.plot(deglist, results['err'])
    #plt.show()

    fig.savefig(os.path.join(directory,"summary.pdf"))
    filename = os.path.join(directory, "periodic-superreg.pkl")
    saveresults(filename, results, noises, datakwargs, mask_kwargs,
                alldata, shifts, img)
