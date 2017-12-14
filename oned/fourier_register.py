import warnings
import numpy as np
import numexpr as ne

from scipy.optimize import golden, leastsq
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter

from leastsq import LM

DEGREE = 20

rng = np.random.RandomState(14850)

class SuperRegistration(object):
    def __init__(self, images, deg, shifts=None, domain=None):
        """
        Parameters
        ----------
        x : ndarray [L]
            List of coordinates for the first image in the sequence.
            It will be used for the basis of all other coordinates for the other images

        y : ndarray [L,N]
            All the images
        """
        self.deg = deg
        self.images = images
        self.shifts = shifts
        self.N = len(self.images)
        self.L = len(self.images[0])

        if self.shifts is None:
            self.shifts = rng.rand(len(self.images)-1)

        self.coef = np.random.randn(2*deg)

        self.x = 1.*np.arange(images[0].shape[0])
        self.k = 2*np.pi*np.arange(self.deg)
        self.domain = domain if domain is not None else [self.x.min(), 
                                                         self.x.max() + self.x.ptp()]

        # two different sums, one for 
        self.sinkx = np.sin(self.k[:,None] * self.coord(self.x)[None,:])
        self.coskx = np.cos(self.k[:,None] * self.coord(self.x)[None,:])

    def set_params(self, params):
        self.shifts = params[:len(self.shifts)]
        self.coef = params[len(self.shifts):]

    @property
    def params(self):
        return np.hstack([self.shifts, self.coef])

    @property
    def model(self):
        return np.array([self(self.x)] + [self(self.x + s) for s in self.shifts])

    @property
    def An(self):
        return self.coef[:self.deg]

    @property
    def Bn(self):
        return self.coef[self.deg:]

    def coord(self, x):
        return (x - self.domain[0]) / (self.domain[1] - self.domain[0])

    def __call__(self, x):
        arg = np.outer(self.coord(x), self.k)
        return (
            np.sin(arg).dot(self.An) +
            np.cos(arg).dot(self.Bn)
        )

    def res(self, params=None):
        params = params if params is not None else self.params
        self.set_params(params)

        return (self.model - self.images).ravel()

    def gradshifts(self, shifts=None):
        shifts = shifts if shifts is not None else self.shifts

        args = self.coord(shifts[:,None] * self.k[None,:])
        sinkd = ne.evaluate('sin(args)')
        coskd = ne.evaluate('cos(args)')

        cn = self.An*self.k
        cm = self.Bn*self.k

        dIds_n = (cn*coskd).dot(self.coskx) - (cn*sinkd).dot(self.sinkx)
        dIds_m = (cm*sinkd).dot(self.coskx) + (cm*coskd).dot(self.sinkx)

        return (dIds_n - dIds_m) / np.diff(self.domain)

    def gradcoef(self):
        allcoords = np.hstack([self.x] + [self.x + s for s in self.shifts])
        allargs = self.k[:,None] * self.coord(allcoords)[None,:]

        sinkx = ne.evaluate('sin(allargs)')
        coskx = ne.evaluate('cos(allargs)')
        return np.vstack([sinkx, coskx])

    def grad(self, params=None):
        if params is not None:
            self.set_params(params)
        else:
            params = self.params

        gcoef = self.gradcoef()
        gshifts = self.gradshifts(self.shifts)

        gradshifts = np.zeros((self.N-1, self.L*self.N))
        for i in range(self.N-1):
            gradshifts[i, (i+1)*self.L:(i+2)*self.L] = gshifts[i]

        return np.vstack([gradshifts, gcoef]).T

    def jac(self, p=None, h=1e-6):
        if p is not None:
            self.set_params(p)
        else:
            p = self.params

        assert len(p) == len(self.params)
        p0 = p.copy()
        r0 = self.res(p0)
        j = []
        for i in range(len(p0)):
            p0[i] += h
            res0 = self.res(p0)
            p0[i] -= 2*h
            res1 = self.res(p0)
            p0[i] += h

            j += [(res0 - res1) / (2*h)]

        return np.array(j).T

    def estimatenoise(self, params=None):
        if params is not None:
            self.set_params(params)
        else:
            params = self.params
        r = self.res(params)
        return np.sqrt(r.dot(r)/len(r))

    def fit(self, images=None, p0=None, **kwargs):
        if images is not None:  # reset images and parameters
            self.images = images
        p0 = p0 if p0 is not None else self.params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lm = LM(self.res, self.grad)
            self.sol = lm.leastsq(p0, **kwargs)
        j = self.grad()
        jtj = j.T.dot(j)
        sigma = self.estimatenoise()

        return self.sol[0][:self.N-1], sigma/np.sqrt(jtj.diagonal()[:self.N-1])


class Fourier(object):
    def __init__(self, x, y, deg, domain=None, coef=None):
        self.y = y
        self.x = x
        self.deg = deg

        if domain is not None:
            self.domain = domain
        else:
            self.domain = Fourier.default_domain(x)

        self.coef = coef if coef is not None else np.random.randn(2*deg - 1)
        assert len(self.coef)==2*self.deg - 1, "Inconsistent deg and coef"
        self.kx = 2*np.pi*np.arange(self.deg)[:,None]

    def __call__(self, x):
        a, b = self.coef[:self.deg,None], self.coef[self.deg:,None]
        arg = self.kx * (x[None,:] - self.domain[0])/np.diff(self.domain)
        return np.sum(a*np.cos(arg), 0) + np.sum(b*np.sin(arg[1:]), 0)

    def res(self, coef):
        self.coef = coef
        return self(self.x)-self.y

    @classmethod
    def fit(cls, x, y, deg, domain=None, **kwargs):
        fourier = cls(x, y, deg, domain)
        coef = kwargs.pop('coef', fourier.coef)
        fourier.coef = leastsq(fourier.res, coef, **kwargs)[0]
        return fourier

    @classmethod
    def default_domain(cls, x, **kwargs):
        r = x.max() - x.min()
        return [x.min()-r, x.min() + r]


class OneDRegister(object):
    def __init__(self, seq, deg, shifts=None, interpolator=Fourier, pad=None,
                 domain=None, **kwargs):
        self.seq = seq
        self.N = len(seq)
        self.L = len(seq[0])
        self.deg = deg
        self.pad = pad
        self.h = kwargs.get('h', 1E-6)
        self.shifts = np.random.rand(self.N-1) if shifts is None else shifts

        self.x = np.arange(self.L)
        self.domain = domain or interpolator.default_domain(self.x, pad=pad)
        self.func = interpolator.fit(
            np.hstack(self.coords), np.hstack(self.seq), self.deg, self.domain
        )

    @property
    def coords(self):
        return np.array([self.x] + [self.x+s for s in self.shifts])

    def getmodel(self, s):
        return self.func(self.x + s)

    def getloglikelihood(self, s, ind):
        model, data = self.getmodel(s), self.seq[ind]
        res = model - data
        return np.dot(res,res)/2.

    @property
    def model(self):
        return np.array(
            [self.func(self.x)] + [self.func(self.x+s) for s in self.shifts]
        )
    
    @property
    def residuals(self):
        return np.array([img - mod for img, mod in zip(self.seq, self.model)])

    @property
    def params(self):
        return np.hstack([self.shifts, self.func.coef])
    
    def res(self, p):
        self.shifts[:] = p[:len(self.shifts)]
        self.func.coef[:] = p[len(self.shifts):]
        return self.residuals.ravel()

    def jac(self, p):
        assert len(p) == len(self.params)
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
        p0 = p0 if p0 is not None else self.params
        lm = LM(self.res, self.jac)
        self.sol = lm.leastsq(p0, **kwargs)
        return self.sol[0][:len(self.shifts)]

def fakedata(N, L, noise, shifts=None, domain=None, coef=None, deg=None):
    shifts = shifts if shifts is not None else rng.rand(N-1)
    coef = coef if coef is not None else rng.randn(2*deg-1)
    x = np.arange(L)
    fourier = Fourier(x, x, deg or (len(coef)+1)/2, coef=coef, domain=domain)
    seq = np.array([fourier(x)] + [fourier(x+s) for s in shifts])
    seq /= seq[0].ptp()
    return seq+noise*rng.randn(*seq.shape), shifts, seq, fourier


if __name__=="__main__":
    from varibayes.infer import VariationalInferenceMF
    degree = 37 # DEGREE
    expt = fakedata(2, 124, 0.05, deg=40)
    data, s, true, fourier = expt

    reg = SuperRegistration(data, 40)
    def loglikelihood(params, data):
        s = params[-1]
        r = (reg.res(params[:-1]) - data)/sigmas
        return - np.sum(r*r + np.sum(2*np.pi*s**2))/2.
    p0 = np.hstack([reg.params/10., 0.05])

    vb = VariationalInferenceMF(loglikelihood, args=(data,))
    vb.fit(p0, iprint=5, itn=1000)
        
