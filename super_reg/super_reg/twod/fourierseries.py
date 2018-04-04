import warnings
import numpy as np
import numexpr as ne
from itertools import chain

from scipy.optimize import golden, leastsq
#from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter

from super_reg.util.leastsq import LM
import super_reg.util.makedata as md

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
            List of 2D coordinates for the first image in the sequence.  It will
            be used for the basis of all other coordinates for the other images

        domain : ndarray [2, 2]
            [[ymin, ymax], [xmin, xmax]]

        """
        self.deg = deg
        self.images = images
        self.images_k = np.fft.rfftn(images, axes=(1,2), norm='ortho')

        self.shifts = shifts
        self.N = len(self.images)
        if self.shifts is None:
            self.shifts = rng.rand(self.N-1, 2)
        self.shape = self.images[0].shape

        self.coef = np.zeros((self.shape[0], self.shape[1]//2+1),
                             dtype='complex128')

        self.ky = np.fft.fftfreq(self.shape[0], d=1./(2*np.pi))[:,None]
        self.kx = np.fft.rfftfreq(self.shape[1], d=1./(2*np.pi))[None,:]

        p0 = np.random.randn(2*(self.N-1) + 2*(deg+1)*(2*deg+1)-1)
        # -1 is to remove imaginary part of zero mode
        self.set_params(p0)
        
    def phase(self, shift):
        return np.exp(-1j*shift[0]*self.ky)*np.exp(-1j*shift[1]*self.kx)

    def set_params(self, params):
        M, deg = 2 * (self.N - 1), self.deg
        self.shifts = params[:M].reshape(self.N-1, 2)
        coefs = params[M:]
        nc = len(coefs)+1  # remove imag zero mode because img is real
        sh = (2*deg+1, deg+1)
        rc, ic = coefs[:nc//2].reshape(sh), coefs[nc//2:]
        ic = np.concatenate((np.array([0.]), ic)).reshape(sh)
        self.coef[:deg+1,:deg+1] = rc[:deg+1] + 1j*ic[:deg+1]
        self.coef[-deg:,:deg+1] = rc[-deg:] + 1j*ic[-deg:]

    def cutout(self, arr, zero=True):
        deg = self.deg
        plus, minus = arr[:deg+1,:deg+1], arr[-deg:,:deg+1]
        cut = np.concatenate((plus, minus), axis=0)
        if zero:
            return np.concatenate((cut.real.ravel(), cut.imag.ravel()))
        else:
            return np.concatenate((cut.real.ravel(), cut.imag.ravel()[1:]))

    @property
    def params(self):
        deg = self.deg
        plus, minus = self.coef[:deg+1,:deg+1], self.coef[-deg:,:deg+1]
        coef = np.concatenate((plus, minus), axis=0)
        return np.concatenate((self.shifts.ravel(), coef.real.ravel(),
                               coef.imag.ravel()[1:]))  #remove imag zero

    @property
    def shiftiter(self):
        return chain([np.zeros(2)], self.shifts)

    @property
    def model(self):
        return np.array([np.fft.irfftn(self.phase(d)*self.coef, norm='ortho')
                         for d in self.shiftiter])

    @property
    def residual(self):
        return self.model - self.images

    @property
    def residual_k(self):
        return np.array([np.abs(dk - self.phase(d)*self.coef)**2 for dk, d in
                         zip(self.images_k, self.shiftiter)])

    def res(self, params=None):
        params = params if params is not None else self.params
        self.set_params(params)

        return np.array([
                self.cutout(dk - self.phase(d)*self.coef, False)
                for dk, d in zip(self.images_k, self.shiftiter)
            ]).ravel()

    def gradshifts(self, params=None):
        params = params if params is not None else self.params
        self.set_params(params)

        jac = []
        cky = 1j*self.ky*self.coef
        ckx = 1j*self.kx*self.coef
        zeros = np.zeros(len(params)-2)
        for n, d in enumerate(self.shifts):
            p = self.phase(d)
            j = np.array([])

            for m, dk in enumerate(self.images_k):
                if n+1 == m:
                    dy = self.cutout(cky*p, False)
                else:
                    dy = zeros
                j = np.concatenate((j, dy))
            jac.append(j)

            j = np.array([])
            for m, dk in enumerate(self.images_k):
                if n+1 == m:
                    dx = self.cutout(ckx*p, False)
                else:
                    dx = zeros
                j = np.concatenate((j, dx))
            jac.append(j)
        
        return np.array(jac).T

    def jac(self, p=None, h=1e-7):
        if p is not None:
            self.set_params(p)
        else:
            p = self.params

        assert len(p) == len(self.params)
        p0 = p.copy()
        r0 = self.res(p0)
        j = []
        for i in range(len(p0)):
            m = p0[i]
            p0[i] = m + h
            res0 = self.res(p0)
            p0[i] = m - h
            res1 = self.res(p0)
            p0[i] = m

            j += [(res0 - res1) / (2*h)]

        return np.array(j).T

    def estimatenoise(self, params=None):
        params = params if params is not None else self.params
        self.set_params(params)
        r = self.residual.ravel()
        return np.sqrt(r.dot(r)/len(r))

    def fit(self, images=None, p0=None, **kwargs):
        if images is not None:  # reset images and parameters
            self.images = images
        p0 = p0 if p0 is not None else self.params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lm = LM(self.res, self.jac)
            self.sol = lm.leastsq(p0, **kwargs)
        j = self.jac()
        jtj = j.T.dot(j)
        sigma = self.estimatenoise()
        shifts = self.shifts
        jtjshifts = jtj.diagonal()[:shifts.size].reshape(*shifts.shape)

        return shifts, sigma/np.sqrt(jtjshifts)


if __name__=="__main__":
    
    import matplotlib.pyplot as plt
    L = 64 
    deg = 12 
    img = md.powerlaw((L, L), 1.8, scale=L/6., rng=rng)
    shifts = np.random.randn(2)
    images = md.fakedata(0., [-shifts], L, img=img, offset=np.zeros(2),
                         mirror=False)

    data = images + 0.05 * np.random.randn(*images.shape)

    reg = SuperRegistration(data, deg)
    s1, s1_sigma = reg.fit(iprint=2,)
