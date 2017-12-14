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
import makedata as md

from onedregister import Register
from fourier_register import SuperRegistration, fakedata

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

def crb(noise, img):
    dimg_k = np.fft.rfft(img)*np.fft.rfftfreq(len(img))
    return noise/np.sqrt(dimg_k.dot(dimg_k.conj()).real)


if __name__=="__main__":
    from datetime import datetime
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    today = datetime.today().isoformat().split("T")[0]

    dataL = 128
    noise = 0.05
    deg = 30
    #img = md.randomblur((3*dataL, 3*dataL), 1., 3.)
    domain=[-3., dataL+3.]
    #domain=[0, dataL]
    img = np.outer(fakedata(2, 3*dataL, 0., deg=deg, domain=domain)[0], np.ones(3*dataL))
    abscissa = np.linspace(0., 2., 10)
    shifts = np.array([[[s, 0.]] for s in abscissa])
    xlabel = "True shift $\Delta_y$"
    padding = 6
    N = 2000

    #directory = 'deltaresults/x-zero-s_{}-p_{}-N_{}-'.format(noise,padding,N)+today

    datakwargs = {'shifts': [np.zeros(2)], 'L': dataL,
                  'sliceobj': np.s_[dataL:2*dataL,dataL+dataL/2],
                  'img': img}
    mask_kwargs = {'sigmoid': {'o': padding, 'w': 0.5, 'o': padding},
                   'constant': {'o': padding, 'l': dataL-2*padding},
                   'linear': {}}

    #try:
    #    os.mkdir(directory)
    #except OSError as oerr:
    #    pass

    fig, axes = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(10,5)) 

    allresults = []
    #for j, mask in enumerate(mask_kwargs):
    biastest = tester.BiasTest(datakwargs, Register, N=N, 
                               mask_kwargs=mask_kwargs['sigmoid'])
    biastest_sr = tester.BiasTest(datakwargs, SuperRegistration, deg,
                                  domain=domain, N=N)

    start = datetime.now()
    alldata = biastest.deltaloop(shifts, noise)
    print("finished fourier shift in {}".format(datetime.now()-start))


    p0 = np.random.randn(len(biastest_sr.reg.params))
    p0 /= np.sqrt(len(p0))

    start = datetime.now()
    alldata_sr = biastest_sr.deltaloop(shifts, noise, p0=p0)
    print("finished super registration in {}".format(datetime.now()-start))

    results = {'bias': [], 'biaserr': [], 'bias_std': [], 'err': []}
    results_sr = {'bias': [], 'biaserr': [], 'bias_std': [], 'err': []}
    for d, dsr, s in zip(alldata, alldata_sr, shifts):
        p1s, p1_sigmas = d
        results['bias'] += [np.mean(p1s)-s[0,0]]
        results['bias_std'] += [np.std(p1s)]
        results['biaserr'] += [np.std(p1s)/np.sqrt(len(p1s))]
        results['err'] += [np.mean(p1_sigmas)]

        p1s, p1_sigmas = dsr
        results_sr['bias'] += [np.mean(p1s)-s[0,0]]
        results_sr['bias_std'] += [np.std(p1s)]
        results_sr['biaserr'] += [np.std(p1s)/np.sqrt(len(p1s))]
        results_sr['err'] += [np.mean(p1_sigmas)]

    f, a = biastest.plotbias(results, abscissa=abscissa, xlabel=xlabel, 
                             axis=axes[0], title="Fourier Registration")
    f, a = biastest.plotbias(results_sr, abscissa=abscissa, xlabel=xlabel, 
                             axis=axes[1], title="Super Registration")
                             #title="{} mask".format(mask))

    crbval = crb(noise, img[datakwargs['sliceobj']])
    axes[0].hlines(crbval, abscissa.min(), abscissa.max(), alpha=0.5,
                   label='CRB')
    axes[0].legend(loc='best')
    axes[1].hlines(crbval, abscissa.min(), abscissa.max(), alpha=0.5,
                   label='CRB')
    axes[0].legend(loc='best')
    
    plt.suptitle("N={}, $\sigma$={}".format(N, noise))
    f.savefig("tests/results/{}-bias-comparison.pdf".format(today))
    #filename = os.path.join(directory,"data-{}_mask-{}.pkl".format(data,mask))
    #with open(filename, 'w') as outfile:
    #    pickle.dump(results, outfile)
    #    pickle.dump(deltas, outfile)
    #    pickle.dump(datakwargs[data], outfile)
    #    pickle.dump(mask_kwargs[mask], outfile)
    #    pickle.dump(alldata, outfile)
    #print("finished {} in {}".format(mask, datetime.now()-start))
    plt.show()
