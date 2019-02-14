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

def evidenceloop(data, sigma, deglist, **kwargs):
    iprint = kwargs.pop('iprint', 0)
    delta = kwargs.pop('delta', 1E-5)
    lamb = kwargs.pop('lamb', .1)
    evdlist = []
    shiftlist = []
    coeflist = []
    
    start = datetime.now()
    for i, d in enumerate(deglist):
        reg = SuperRegistration(data, d, **kwargs)
        shiftlist.append(reg.fit(iprint=iprint, delta=delta, lamb=lamb))
        evdlist.append(reg.evidence(sigma=sigma))
        coeflist.append(reg.coef)
        if iprint:
            print("d = {} evd = {}".format(d, evdlist[-1]))
    if iprint:
        print("Calculation took {}".format(datetime.now()-start))
    return evdlist, shiftlist, coeflist    

def standardmethod(data, **kwargs):
    fshift = util.firsttry(data, **kwargs)
    fsrecon = util.shiftallto(data, fshift)
    return fshift, fsrecon

def noiseloop(sigma, model, params, pnames, p, M=20, deglist=range(5,20),
              **kwargs):
    iprint = kwargs.pop('iprint', 0)
    delta = kwargs.pop('delta', 1E-5)
    lamb = kwargs.pop('lamb', .1)

    # find best evidence
    data = model + sigma * rng.randn(*model.shape)
    evdlist, shiftlist, coeflist = evidenceloop(data, sigma, deglist,
                                                iprint=iprint)
    # warn if peak is at edges
    argmax = np.argmax(evdlist)
    maxdeg = list(deglist)[argmax]
    if argmax == 0 or argmax == len(evdlist)-1:
        print("Peak evidence is at the edge of deglist")
    # loop over noise
    sr_results = []
    fs_results = []
    print("\tStarting noise loop with deg={}".format(maxdeg))
    for i in range(M):
        data = model + sigma * rng.randn(*model.shape)

        reg = SuperRegistration(data, maxdeg)
        reg.fit(iprint=iprint, delta=delta, lamb=lamb)
        srmog = makemog(reg.model[0], sigma, *params)
        sr_results.append(srmog.fit(
            pnames, p, delta=1E-6, iprint=iprint, solver='spsolve'
        )[:2])

        fshift, fsrecon = standardmethod(data)
        fsmog = makemog(fsrecon, sigma, *params)
        fs_results.append(fsmog.fit(
            pnames, p, delta=1E-6, iprint=iprint, solver='spsolve'
        )[:2])

    return maxdeg, sr_results, fs_results


if __name__ == '__main__':
    #Experimental setup
    M = 1000  # number of noise instances
    Nsigma = 20  # number of noise strengths

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
    truep = tmog.fit(pnames, p0, delta=1E-6, iprint=1, solver='cg')

    sigmalist = np.linspace(0.01, .3, Nsigma)
    results = []
    maxdeglist = []

    start = datetime.now()
    for i, sigma in enumerate(sigmalist):
        print("Starting sigma={}".format(sigma))
        deglist = range(maxdeg-2, maxdeg+3) if i else range(12, 16)

        maxdeg, sr_results, fs_results = noiseloop(
            sigma, model, params, pnames, p0, M=M, deglist=deglist,
        )
        maxdeglist.append(maxdeg)
        results.append([sr_results, fs_results])
    print("Calculation finished in {}".format(datetime.now()-start))

    savename = '{}-twop-M_{}-N_{}-Nsigma_{}-L_{}'.format(today, M, N, Nsigma, L)
    saveloc = os.path.join('data', savename)
    np.savez(saveloc, M=M, Nsigma=Nsigma, N=N, L=L, shifts=shifts, s0=s0,
             pos=pos, widths=widths, params=params, model=model,
             sigmalist=sigmalist, results=results, truep=truep, pnames=pnames)
