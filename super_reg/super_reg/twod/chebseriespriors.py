import warnings
import numpy as np
import numexpr as ne
from itertools import chain

from matplotlib.pyplot import *
from numpy.polynomial.chebyshev import chebval, chebval2d
from scipy.linalg import svdvals, solve

from super_reg.util.leastsq import LM
import super_reg.util.makedata as md
from super_reg.twod.chebseries import SuperRegistration

DEGREE = 20

rng = np.random.RandomState(14850)


class SuperRegistrationPriors(SuperRegistration):
    def __init__(self, images, deg, shifts=None, domain=None, gamma=None):
        self.gamma = gamma if gamma is not None else 0.

        degs = np.arange(deg+1)
        self.mplusn = degs[:,None] + degs[None,:]
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
        return np.sqrt((self.gamma * self.mplusn).ravel())

    def logpriors(self, params=None, sigma=None):
        params = params if params is not None else self.params
        sigma = 1 if sigma is None else sigma
        M = 2 * (self.N - 1)
        return sigma * self.gmat * params[M:]

    def gradlogpriors(self, params=None, sigma=None):
        params = params if params is not None else self.params
        sigma = 1 if sigma is None else sigma
        S, N = self.shifts.size, self.images.size
        self.glogp[:, S:] = sigma * np.diagflat(self.gmat)
        return self.glogp

    def estimatecoefs(self):
        tmats = [self.tmatrix(s) for s in self.shiftiter]
        A = np.sum([t.T.dot(t) for t in tmats], 0)
        b = np.sum([t.T.dot(d.ravel()) for t, d in zip(tmats, self.images)], 0)
        return solve(A + np.diagflat(self.gmat**2), b).reshape(self.deg+1,-1)

    def evidenceparts(self, sigma=None):
        s = sigma if sigma is not None else self.estimatenoise()
        r = self.res()
        N = len(self.params)
        j = self.gradposterior(sigma=s)
        logdet = 2*np.log(svdvals(j)).sum()
        jtr = j[:self.images.size].T.dot(r)
        rcorrection = jtr.T.dot(solve(j.T.dot(j), jtr))/s**2
        return np.array([-r.dot(r)/s**2, N*np.log(2*np.pi*s**2), -logdet,
                         rcorrection])/2.


if __name__=="__main__":
    
    from scipy.misc import face
    import super_reg.twod.fourierseries as fs
    import matplotlib.pyplot as plt

    deg = 8
    L = 32
    img = md.powerlaw((2*L, 2*L), 1.8, scale=2*L/6., rng=rng)
    shifts = rng.randn(2)
    images = md.fakedata(0., [shifts], L, img=img, offset=L*np.ones(2),
                         mirror=False)
    images /= images.ptp()

    sigma = 0.025
    data = images + sigma * rng.randn(*images.shape)

    reg = SRGaussianPriors(data, 18, gamma=.1)
    #s1, s1s = reg.fit(iprint=1, delta=1E-8, maxiter=100)
    
    if False:
        gammalist = np.logspace(-2., 1.5, num=20)
        degree = 30
        evd = []
        costs = []
        ans = []
        ans_sigma = []
        for g in gammalist:
            reg = SRGaussianPriors(data, degree, gamma=g)
            s1, s1s = reg.fit(iprint=0, delta=1E-6, itnlim=100)
            r = reg.res()
            nlnprob = r.T.dot(r)/2.

            costs.append(nlnprob)
            evd.append(reg.evidenceparts(sigma=sigma))
            ans.append(s1.squeeze())
            ans_sigma.append(s1s.squeeze())
            print("Finished g={} with evd={}".format(g, evd[-1].sum()))

        evd = np.array(evd)
        ans = np.array(ans)
        ans_sigma = np.array(ans_sigma)
