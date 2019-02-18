"""
chebseriespriors.py

author: Colin Clement
date: 2018-05-09

This module extends chebseries.SuperRegistration to put prior
constraints on the image model Chebyshev coefficients.
"""

import warnings
import numpy as np
import numexpr as ne
from itertools import chain

from matplotlib.pyplot import *
from numpy.polynomial.chebyshev import chebval, chebval2d
from scipy.linalg import svdvals, solve
from scipy.optimize import fminbound

from superreg.util.leastsq import LM
import superreg.util.makedata as md
from superreg.chebseries import SuperRegistration


class SuperRegistrationPriors(SuperRegistration):
    def __init__(self, images, deg, shifts=None, domain=None, gamma=None):
        self.gamma = gamma if gamma is not None else 0.

        degs = np.arange(deg+1)
        self.mdist = 1E-6 + np.hypot(degs[:,None], degs[None,:])
        S = 2 * (len(images)-1)
        M, N = (deg+1)**2+S, images.size
        self.glogp = np.zeros((M-S, M))
        super().__init__(images, deg, shifts, domain)

    def logpriors(self, params=None):
        pass

    def gradlogpriors(self, params=None):
        pass

    def gammafunc(self, params=None):
        pass

    def resposterior(self, params=None, sigma=None):
        return np.concatenate([self.res(params), 
                               self.logpriors(params, sigma)])

    def gradposterior(self, params=None, h=1E-6, sigma=None):
        return np.concatenate([self.grad(params, h), 
                               self.gradlogpriors(params, sigma)])
    
    def jacposterior(self, p=None, h=1e-6):
        if p is not None:
            self.set_params(p)
        else:
            p = self.params

        assert len(p) == len(self.params)
        p0 = p.copy()
        r0 = self.res(p0)
        j = []
        for i in range(len(p0)):
            c = p0[i]
            p0[i] = c + h
            res0 = self.resposterior(p0)
            p0[i] = c - h
            res1 = self.resposterior(p0)
            p0[i] = c

            j += [(res0 - res1) / (2*h)]

        return np.array(j).T

    def setoptimizer(self):
        self.opt = LM(self.resposterior, self.gradposterior)


class SRGaussianPriors(SuperRegistrationPriors):
    @property
    def gmat(self):
        L = max(self.images.shape[1:])
        shiftlp = np.ones(self.shifts.size)/(10*L)
        g = np.sqrt(self.gamma)
        return g*np.concatenate([shiftlp, np.sqrt(self.mdist.ravel())])

    def logpriors(self, params=None, sigma=None):
        params = params if params is not None else self.params
        sigma = 1 if sigma is None else sigma
        return sigma * self.gmat * params

    def gradlogpriors(self, params=None, sigma=None):
        params = params if params is not None else self.params
        sigma = 1 if sigma is None else sigma
        S, N = self.shifts.size, self.images.size
        self.glogp = sigma * np.diagflat(self.gmat)
        return self.glogp

    def bestcoef(self):
        tmats = [self.tmatrix(s) for s in self.shiftiter]
        A = np.sum([t.T.dot(t) for t in tmats], 0)
        b = np.sum([t.T.dot(d.ravel()) for t, d in zip(tmats, self.images)], 0)
        gmat = self.gmat[2 * (self.N - 1):]
        return solve(A + np.diagflat(gmat**2), b).reshape(self.deg+1,-1)

    def paramerrors(self, params=None, sigma=None):
        sigma = sigma if sigma is not None else self.estimatenoise()
        j = self.gradposterior(params)
        jtj = j.T.dot(j)
        return sigma/np.sqrt(jtj.diagonal())

    def cost(self, params=None):
        return np.sum(self.resposterior(params)**2)/2.

    def evidenceparts(self, sigma=None):
        s = sigma if sigma is not None else self.estimatenoise()
        r = self.res()
        lp = self.logpriors()
        N = len(self.params)
        j = self.gradposterior(sigma=s)/s**2  # JTJ/s**2 + GTG
        logdetA = 2*np.log(svdvals(j)).sum()
        logdetgtg = np.log(self.gmat**2).sum()
        return np.array([-r.dot(r)/s**2, -lp.dot(lp), -N*np.log(2*np.pi*s**2), 
                         -logdetA, logdetgtg])/2.

    def optevidence(self, sigma=None, **kwargs):
        sigma = sigma if sigma is not None else self.estimatenoise()
        iprint = kwargs.pop('iprint', 0)
        delta = kwargs.pop('delta', 1E-4)
        tol = kwargs.pop('tol', 1E-2)
        bound = kwargs.pop('bound', (0., 1000))
        p0 = self.params
        def minusevd(gamma):
            self.gamma = gamma
            if iprint:
                print("\tgamma={}".format(gamma))
            self.set_params(p0.copy())
            self.fit(iprint=iprint, delta=delta)
            return -self.evidence(sigma)
        return fminbound(minusevd, bound[0], bound[1], xtol=tol, **kwargs)


def optcomplexity(data, registration, sigma=None, gamma=None, **kwargs):
    d0 = kwargs.pop('d0', 5)
    g0 = gamma if gamma is not None else kwargs.pop('g0', 1.)
    show = kwargs.pop('show', False)
    reg = registration(data, d0, gamma=g0)
    g0 = reg.optevidence(sigma, **kwargs)
    e0 = reg.evidenceparts(sigma).sum()
    converged = False
    while not converged:
        if show:
            print("d={}, evd={}, g={}".format(d0, e0, g0))
        reg = registration(data, d0+1, gamma=g0)
        g1 = gamma if gamma is not None else reg.optevidence(sigma, **kwargs)
        e1 = reg.evidenceparts(sigma).sum()
        if e1 < e0:  # if evidence decreases with increasing complexity
            converged = True
        else:
            d0 += 1
            g0, e0 = g1, e1
    return d0, g0
       
        

if __name__=="__main__":
    
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(14850)

    degree = 20
    L = 32
    img = md.powerlaw((2*L, 2*L), 1.8, scale=2*L/6., rng=rng)
    shifts = rng.randn(2)
    images = md.fakedata(0., [shifts], L, img=img, offset=L*np.ones(2),
                         mirror=False)
    images /= images.ptp()

    sigma = 0.05
    data = images + sigma * rng.randn(*images.shape)
    evdloop = False

    reg = SRGaussianPriors(data, degree, gamma=1)
    p = reg.params.copy()
    if not evdloop:
        pass
        #s1, s1s = reg.fit(iprint=1, delta=1E-8, maxiter=100)
        #reg.set_params(p)
        #s1i, s1si = reg.itnfit(iprint=0, delta=1E-8, maxiter=100)
    
    if evdloop:
        gammalist = np.logspace(.2, 1.5, num=10)
        deglist = np.arange(5, 15)
        evd = []
        bestgammas = []
        costs = []
        ans = []
        for d in deglist:
            reg = SRGaussianPriors(data, d, gamma=1.)
            bestgamma = reg.optevidence(sigma, iprint=0, tol=1E-3, delta=1E-5,)
            r = reg.res()
            nlnprob = r.T.dot(r)/2.

            costs.append(nlnprob)
            evd.append(reg.evidenceparts(sigma=sigma))
            bestgammas.append(bestgamma)
            ans.append(reg.shifts.squeeze())
            print("Finished d={} evd={:.1f}, g={:.2f}".format(d, evd[-1].sum(),
                                                              bestgamma))

        evd = np.array(evd)
        bestgammas = np.array(bestgammas)
        ans = np.array(ans)
