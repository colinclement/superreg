import numpy as np
from scipy.optimize import golden, leastsq
from matplotlib.pyplot import *
from numpy.polynomial.chebyshev import Chebyshev
from scipy.ndimage import gaussian_filter

DEGREE = 20

rng = np.random.RandomState(14850)

class OneDRegister(object):
    def __init__(self, seq, deg, shifts=None, **kwargs):
        self.seq = seq
        self.N = len(seq)
        self.L = len(seq[0])
        self.deg = deg
        self.pad = kwargs.get('p', 2)
        self.h = kwargs.get('h', 1E-6)
        self.domain = kwargs.get('domain', [-self.pad, self.L+self.pad])

        self.x = np.arange(self.L)
        self.shifts = np.zeros(self.N-1) if shifts is None else shifts
        self.cheb = Chebyshev.fit(
            np.hstack(self.coords), np.hstack(self.seq), self.deg, self.domain
        )

    @property
    def coords(self):
        return np.array([self.x] + [self.x+s for s in self.shifts])

    def getmodel(self, s):
        return self.cheb(self.x + s)

    def getloglikelihood(self, s, ind):
        model, data = self.getmodel(s), self.seq[ind]
        res = model - data
        return np.dot(res,res)/2.

    @property
    def model(self):
        return np.array(
            [self.cheb(self.x)] + [self.cheb(self.x+s) for s in self.shifts]
        )
    
    @property
    def residuals(self):
        return np.array([img - mod for img, mod in zip(self.seq, self.model)])

    @property
    def p(self):
        return np.hstack([self.shifts, self.cheb.coef])
    
    def res(self, p):
        self.shifts[:] = p[:len(self.shifts)]
        self.cheb.coef[:] = p[len(self.shifts):]
        return self.residuals.ravel()

    def jac(self, p):
        assert len(p) == len(self.p)
        p0 = p.copy()
        r0 = self.res(p0)
        j = []
        for i in range(len(p0)):
            p0[i] += self.h
            j += [(self.res(p0) - r0)/self.h]
            p0[i] -= self.h
        return np.array(j).T

    @property
    def loglikelihood(self):
        residuals = self.residuals
        return np.dot(residuals.flat, residuals.flat)

    def optimize(self, shifts=None):
        shifts = shifts if shifts is not None else self.shifts
        newshifts = np.zeros_like(shifts)
        for i, s in enumerate(shifts):
            newshifts[i] = golden(self.getloglikelihood, args=(i+1,))
        return newshifts

    def fit(self, p0=None, **kwargs):
        p0 = p0 if p0 is not None else self.p
        self.sol = leastsq(self.res, p0, **kwargs)
        return self.sol[0][:len(self.shifts)]

def fakedata(N, L, noise, shifts=None, scale=6., deg=DEGREE):
    shifts = shifts if shifts is not None else rng.rand(N-1)
    x = np.arange(L)
    rnd = gaussian_filter(rng.randn(2*L), sigma=scale)
    cheb = Chebyshev.fit(np.arange(-L/2., 3.*L/2), rnd, deg, domain=[-L/2., 3.*L/2])
    seq = np.array([cheb(x)] + [cheb(x+s) for s in shifts])
    seq /= seq[0].ptp()
    return seq+noise*rng.randn(*seq.shape), shifts, seq, cheb

def fit(images, deg=DEGREE, itn=10, shifts=None, iprint=True):
    reg = OneDRegister(images, deg=deg, shifts=shifts)
    for i in xrange(itn):
        if iprint:
            print(reg.shifts, reg.loglikelihood)
        reg = OneDRegister(images, deg=reg.deg, shifts=reg.shifts)
        reg.shifts = reg.optimize()
    return reg

def evaluatebias(num=200, N=2, L=128, noise=0.01, itn=5, deg=20, expt=None):
    data, s, true, cheb = expt or fakedata(N, L, noise)
    results = []
    models = []
    for i in range(num):
        d = true + noise*rng.randn(*true.shape)
        reg = OneDRegister(d, deg, shifts = np.array([rng.rand()])) 
        sol = reg.fit()
        results += [sol]
        models += [reg]
    bias = np.mean(np.array(results)-s)
    bias_std = np.std(np.array(results)-2)/np.sqrt(num)
    return bias, bias_std, np.array(results), reg, models
    

if __name__=="__main__":
    degree = DEGREE
    expt = fakedata(16, 100, 0.01, scale=6., deg=degree)
    data, s, true, cheb = expt
    reg = OneDRegister(data, deg=degree)

    #biases = []
    #biases_std = []
    #for n in np.linspace(0.01, 0.15, 20):
    #    bias, bias_std, results, reg, models = evaluatebias(100, noise=n, expt=expt) 
    #    biases += [bias]
    #    biases_std += [bias_std]
    #    print(n, bias, bias_std)
    #errorbar(n, biases, yerr=biases_std)
