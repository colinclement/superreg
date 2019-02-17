"""
fourierseries.py

author: Colin Clement and Matt Bierbaum
date: 2018-02-01

This module performs image registration on periodic images by
modeling the underlying true image as a sum of Fourier components.
"""


import warnings
import numpy as np
import numexpr as ne
from itertools import chain

from scipy.optimize import golden, leastsq
from scipy.ndimage import gaussian_filter
from numpy.linalg import slogdet

from superreg.util.leastsq import LM
import superreg.util.makedata as md

rng = np.random.RandomState(14850)


class SuperRegistration(object):
    def __init__(self, images, deg, shifts=None, domain=None, gamma=0.):
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

        gamma : float
            Prior strength penalizing magnitude of Fourier coefficients
        """
        Ly, Lx = images[0].shape
        self.deg = min(deg, Ly//2, Lx//2)
        self.images = images
        self.images_k = np.fft.rfftn(images, axes=(1,2), norm='ortho')
        self.gamma = gamma

        self.shifts = shifts
        self.N = len(self.images)
        if self.shifts is None:
            self.shifts = rng.rand(self.N-1, 2)
        self.shape = self.images[0].shape

        self.coef = np.zeros((self.shape[0], self.shape[1]//2+1),
                             dtype='complex128')

        self.ky = np.fft.fftfreq(self.shape[0], d=1./(2*np.pi))[:,None]
        self.kx = np.fft.rfftfreq(self.shape[1], d=1./(2*np.pi))[None,:]

        self.p0 = np.random.randn(2*(self.N-1) + 2*(deg+1)*(2*deg+1)-1)
        # -1 is to remove imaginary part of zero mode
        self.set_params(self.p0)
        
    def phase(self, shift):
        """ e^(i k shift) """
        return np.exp(-1j*shift[0]*self.ky)*np.exp(-1j*shift[1]*self.kx)

    def set_params(self, params):
        """
        Fill parameters into coef matrix so that image generation can
        be performed via a single Fourier Transform
        Parameters
        ----------
        params : array_like
            list of fourier coefficients, real then imaginary
        """
        M, deg = 2 * (self.N - 1), self.deg
        self.shifts = params[:M].reshape(self.N-1, 2)
        coefs = params[M:]
        nc = len(coefs)+1  # remove imag zero mode because img is real
        sh = (2*deg+1, deg+1)
        rc, ic = coefs[:nc//2].reshape(sh), coefs[nc//2:]
        ic = np.concatenate((np.array([0.]), ic)).reshape(sh)
        self.coef[:deg+1,:deg+1] = rc[:deg+1] + 1j*ic[:deg+1]
        self.coef[-deg:,:deg+1] = rc[-deg:] + 1j*ic[-deg:]

    def slice(self, arr):
        deg = self.deg
        return np.concatenate((arr[:deg+1,:deg+1], arr[-deg:,:deg+1]), axis=0)

    def comp(self, arr, zero=False):
        if zero:
            return np.concatenate((arr.real.ravel(), arr.imag.ravel()))
        else:
            return np.concatenate((arr.real.ravel(), arr.imag.ravel()[1:]))

    def cutout(self, arr, zero=False):
        cut = self.slice(arr)
        return self.comp(cut, zero)

    @property
    def params(self):
        return np.concatenate((self.shifts.ravel(),
                               self.comp(self.slice(self.coef), zero=False)))

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

        c = self.slice(self.coef)
        r = np.array([
                self.comp(self.slice(dk) - self.slice(self.phase(d))*c)
                for dk, d in zip(self.images_k, self.shiftiter)
            ]).ravel()
        if self.gamma > 0:
            r = np.concatenate((r, np.sqrt(self.gamma)*self.comp(c)))
        return r

    def gradshifts(self, params=None):
        params = params if params is not None else self.params
        self.set_params(params)

        jac = []
        cky = 1j*self.ky*self.coef
        ckx = 1j*self.kx*self.coef
        zeros = np.zeros(len(params)-2, dtype='float64')
        for n, d in enumerate(self.shifts):
            p = self.phase(d)
            j = np.array([])

            for m, dk in enumerate(self.images_k):
                if n+1 == m:
                    dy = self.cutout(cky*p, False)
                else:
                    dy = zeros
                j = np.concatenate((j, dy))
            if self.gamma > 0:
                j = np.concatenate((j, zeros))
            jac.append(j)

            j = np.array([])
            for m, dk in enumerate(self.images_k):
                if n+1 == m:
                    dx = self.cutout(ckx*p, False)
                else:
                    dx = zeros
                j = np.concatenate((j, dx))
            if self.gamma > 0:
                j = np.concatenate((j, zeros))
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

    def lmfit(self, images=None, p0=None, sigma=None, **kwargs):
        if images is not None:  # reset images and parameters
            self.images = images
            self.images_k = np.fft.rfftn(images, axes=(1,2), norm='ortho')
        p0 = p0 if p0 is not None else self.params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lm = LM(self.res, self.jac)
            self.sol = lm.leastsq(p0, **kwargs)
        j = self.jac()
        jtj = j.T.dot(j)
        sigma = self.estimatenoise() if sigma is None else sigma
        shifts = self.shifts
        jtjshifts = jtj.diagonal()[:shifts.size].reshape(*shifts.shape)

        return shifts, sigma/np.sqrt(jtjshifts)

    def averagecoefs(self):
        """
        Estimate of the best coefficients is the average of all the inverse
        shifted data
        """
        N = self.images.shape[0]
        return np.array([
            self.comp(self.slice(dk) * self.slice(self.phase(-d)))
            for dk, d in zip(self.images_k, self.shiftiter)
        ]).sum(0)/(N+self.gamma)

    def shiftres(self, shifts=None):
        if shifts is not None:
            self.shifts = shifts.reshape(*self.shifts.shape)
        return self.res()

    def shiftgrad(self, shifts=None):
        if shifts is not None:
            self.shifts = shifts.reshape(*self.shifts.shape)
        return self.gradshifts()

    def minshift(self, shifts0=None, **kwargs):
        shifts0 = shifts0 if shifts0 is not None else self.shifts.ravel()
        lm = LM(self.shiftres, self.shiftgrad)
        if 'tol' in kwargs:
            kwargs['tol'] = 1E-2
        self.shiftsol = lm.leastsq(shifts0, **kwargs)
        return self.shiftsol[0]

    def fit(self, images=None, p0=None, sigma=None, abschange=1E-6, **kwargs):
        """
        Fit the image model and shifts
        Parameters
        ----------
        images : array_like
            stack of images
        p0 : array_like
            initial parameter values of Fourier coefficients
        sigma : float
            estimate of noise in images
        abschange : float
            convergence criterion for alternating optimization
        Returns
        -------
            shifts, shift_uncertainties
        """
        if images is not None:  # reset images and parameters
            self.images = images
            self.images_k = np.fft.rfftn(images, axes=(1,2), norm='ortho')
        if p0 is not None:
            self.set_params(p0)
        else:
            p0 = self.params
        iprint = kwargs.get('iprint', 0)

        r = self.res()
        c0 = r.T.dot(r)/2.
        converged, i = False, 0
        while not converged:
            c = self.averagecoefs() 
            self.set_params(np.concatenate((self.shifts.ravel(), c)))
            s = self.minshift() # **kwargs)

            r = self.res()
            c1 = r.T.dot(r)/2.
            if np.abs((c1 - c0)) < abschange:
                converged = True
            if iprint:
                print("Outer Itn {} nlnprob = {}".format(i, c1))
            c0 = c1
            i += 1
        j = self.gradshifts()
        jtj = j.T.dot(j).diagonal()
        sigma = self.estimatenoise() if sigma is None else sigma
        var = sigma/np.sqrt(jtj)
        return self.shifts, var.reshape(*self.shifts.shape)

    def evidence(self, sigma=None):
        """ 
        Calculate the log-evidence or estimate of the posterior
        normalization to perform model selection
        Parameters
        ----------
        sigma : float
            Noise level of images, estimated by other means if not provided
        Returns
        -------
        log_evidence : float
        """
        s = self.estimatenoise() if sigma is None else sigma
        r = np.concatenate((self.residual.ravel(), 
                            s*np.sqrt(self.gamma)*np.abs(self.coef).ravel()))
        n = len(self.params)
        N = self.images.shape[0]
        j = self.gradshifts()
        jtj = j.T.dot(j).diagonal()
        logdet = np.log(jtj).sum() + (n-len(jtj))*np.log(N+self.gamma)
        # logdet is calculated by examining structure of jtj
        return (-r.dot(r)/s**2 + n*np.log(2*np.pi*s**2) - logdet)/2.


if __name__=="__main__":
    
    def crb(img, deg=None):
        Ly, Lx = img.shape
        degy = deg if deg is not None else Ly
        degx = deg if deg is not None else Lx
        ky = np.fft.fftfreq(Ly, d=1./(2*np.pi))
        ky[degy:] = 0.
        kx = np.fft.fftfreq(Lx, d=1./(2*np.pi))
        kx[degx:] = 0.
        Ik = np.fft.fftn(img, norm='ortho')
        Ik2 = Ik*Ik.conj()
        return np.sum(ky**2*Ik2).real, np.sum(kx**2*Ik2).real
    
    import matplotlib.pyplot as plt
    L = 128
    deg = L//2
    sigma = 0.1
    img = md.powerlaw((L, L), 1.8, scale=L/6., rng=rng)
    shifts = np.random.randn(2)
    images = md.fakedata(0., [-shifts], L, img=img, offset=np.zeros(2),
                         mirror=False)

    data = images + sigma * np.random.randn(*images.shape)

    reg = SuperRegistration(data, deg, gamma=1.15)
    reg2 = SuperRegistration(data, deg, gamma=.1)
    reg2.set_params(reg.params.copy())

    #s1, s1_sigma = reg.fit(iprint=1, sigma=sigma)
    #s1, s1_sigma = reg2.lmfit(iprint=1, tol=1E-8, sigma=sigma)
    s1_crb = sigma/np.sqrt(crb(img))

    evd, cost, ests = [], [], []
    orders = range(6, 25)
    #gammas = 10**np.linspace(0., .5, num=40)-1
    gammas = np.linspace(0., 4., num=40)
    results = []
    
    # deg = 19 was peak for img scale 1.8, L=128 and scale=L/6.
    for deg in orders:
    #for g in gammas:
        reg = SuperRegistration(data, deg, gamma=0.)
        reg.fit(iprint=0)
        r = reg.res()
        cost.append(0.5*r.T.dot(r))
        ests.append(reg.estimatenoise())
        results.append(reg.shifts.copy().squeeze())
        evd.append(reg.evidence2(sigma=sigma))
        #print("Fit gamma={:.4f} with evd={:.1f} c={:.1f}".format(g, evd[-1],
        #                                                    cost[-1]))
        print("Fit deg={} with evd={:.1f} c={:.1f}".format(deg, evd[-1],
                                                            cost[-1]))
