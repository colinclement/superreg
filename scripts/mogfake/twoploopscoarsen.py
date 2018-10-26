import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from super_reg.twod.chebseries import SuperRegistration
from super_reg.twod.fouriershift import Register
import makedata as md

os.sys.path.append('../')
import util
from mog import mog

mpl.rcParams['image.cmap'] = 'Greys'
today = format(datetime.now()).split()[0]
rng = np.random.RandomState(92089)

def makemog(data, sigma, pos, widths):
    Ly, Lx = data.shape
    m = mog.Mog(data, boxprior=0., anisprior=0.)
    p0 = np.array([2.*Ly, Ly*pos[0][1], -Ly*pos[0][0], Ly*widths[0],
                   2.*Ly, Ly*pos[1][1], -Ly*pos[1][0], Ly*widths[1],
                   0., 1., sigma, 0.]).astype('float')
    pnames = []
    pnames += [n.format(0) for n in m.localnameformat]
    pnames += [n.format(1) for n in m.localnameformat]
    pnames += m.globalnames
    m.register(pnames, p0)
    return m

def makemodel(shifts, pos, widths, **kwargs):
    return md.twoparticles(np.concatenate([[np.zeros(2)], shifts]), pos=pos,
                           widths=widths)

def standardmethod(data, a=1, **kwargs):
    fshift = util.coarsereg(data, a, **kwargs)
    fsrecon = util.shiftallto(data, fshift)
    return fshift, fsrecon

def noiseloop(sigma, model, params, pnames, p, M=20, alist=[1, 2, 3, 4],
              **kwargs):
    iprint = kwargs.pop('iprint', 0)
    delta = kwargs.pop('delta', 1E-5)
    lamb = kwargs.pop('lamb', .1)

    fs_results = []
    for i in range(M):
        data = model + sigma * rng.randn(*model.shape)
        ares = []
        for a in alist:
            fshift, fsrecon = standardmethod(data, a)
            fsmog = makemog(fsrecon, sigma, *params)
            ares.append(fsmog.fit(
                pnames, p, delta=1E-6, iprint=iprint
            )[0])  # only keep best fits
        fs_results.append(ares)

    return fs_results


if __name__ == '__main__':
    #Experimental setup
    M = 1000  # number of noise instances
    Nsigma = 20  # number of noise strengths
    alist = [1, 2, 3, 4]
    N = 8
    L = 64

    s0 = rng.randn(N-1, 2)/(L/2)
    shifts = L * s0
    pos = [0.3*rng.rand(2)+0.0, 0.4*rng.rand(2)+0.5]
    widths = [0.1, 0.16]
    params = (pos, widths)
    model = makemodel(s0, pos, widths)

    tmog = makemog(model[0], 0.01, *params)
    pnames = list(tmog.pmap.names)
    pnames.remove('background')
    pnames.remove('epsilon')
    pnames.remove('theta')
    p0 = tmog.pmap.getvalues(pnames)
    truep = tmog.fit(pnames, p0, delta=1E-6)

    sigmalist = np.linspace(0.01, .3, Nsigma)
    results = []

    start = datetime.now()
    for i, sigma in enumerate(sigmalist):
        print("Starting sigma={}".format(sigma))

        fs_results = noiseloop(
            sigma, model, params, pnames, p0, M=M, alist=alist
        )
        results.append(fs_results)
    print("Calculation finished in {}".format(datetime.now()-start))

    savename = '{}-twop-coarsen-M_{}-N_{}-Nsigma_{}-L_{}'.format(today, M, N, Nsigma, L)
    saveloc = os.path.join('data', savename)
    np.savez(saveloc, M=M, Nsigma=Nsigma, N=N, L=L, shifts=shifts, s0=s0,
             pos=pos, widths=widths, params=params, model=model, alist=alist,
             sigmalist=sigmalist, results=results, truep=truep, pnames=pnames)
