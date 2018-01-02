import warnings
import numpy as np
import numexpr as ne

from scipy.optimize import golden, leastsq
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter

from super_reg.util.leastsq import LM

DEGREE = 20

rng = np.random.RandomState(14850)


class SuperRegistration(object):
    def __init__(self, images, deg, shifts=None, domain=None):
        """
        Parameters
        ----------
        images : ndarray [N, L, L]
            All the images

        deg : int
            Maximum degree of fourier mode

        shifts : ndarray [N - 1, 2]
            List of 2D coordinates for the first image in the sequence.
            It will be used for the basis of all other coordinates for the other images

        domain : ndarray [2, 2]
            [[ymin, ymax], [xmin, xmax]]

        """
        self.deg = deg
        self.images = images
        self.shifts = shifts
        self.N = len(self.images)
        self.L = len(self.images[0])

        if self.shifts is None:
            self.shifts = rng.rand(len(self.images)-1)

        self.coef = np.random.randn(2*deg, 2*deg)

        self.x = 1.*np.arange(images[0].shape[0])
        self.y = 1.*np.arange(iamges[0].shape[1])[:,None]

        self.kx = 2*np.pi*np.arange(self.deg)
        self.ky = self.ky[:,None]

        if domain is not None:
            self.domain = domain
        else:
            self.domain = np.array(
                [[self.y.min(), self.y.min() + self.y.ptp()],
                 [self.x.min(), self.x.max() + self.x.ptp()]]
            )

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
        
