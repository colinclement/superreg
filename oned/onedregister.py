"""
register.py

author: Colin Clement
date: 2017-06-06

This module computes the registration error between two images, i.e. it finds a shift
which most matches two images by finding an optimal shift to match the two.
"""

import numpy as np
try:
    from fftw import FFT
    hasfftw = True
except ImportError as ierr:
    print("Install pyfftw for 20x speedup")
    hasfftw = False
import leastsq
from scipy.optimize import minimize_scalar
from scipy.special import expit


class Register(object):
    """
    Given some underlying function f(y, x), two images are
    sampled I0 = f(y, x) + xi and I1 = f(y+dy, x+dx) + eta
    which differ by some translation delta = [dy, dx] and have different
    experimental noise xi and eta. Given I0 and I1 we want to infer delta.

    Define T such that T_delta(I) translates image I by delta. Then to 
    infer delta we want to solve the following optimization problem:
        delta^* = min_delta 1/2 ||I0 - T_delta(I1)||^2
    """
    def __init__(self, images, **kwargs):
        """
        Find the vector [dy, dx] which translates imag1 to match imag0.

        Inputs:
            (required)
            imag1: array_like image of shape (Ly, Lx)
            imag0 : array_like image of shape (Ly, Lx)
        kwargs:
            h : float, finite difference step size for estimating jacobian
            all other kwargs are passed into FFT (above)

        usage:
            reg = Register(imag1, imag0)
            delta, delta_sigma, msg = reg.fit()
        """
        self.initialize(images)
        assert self.imag0.shape == self.imag1.shape, "Images must share their shape"
        self.masktype = kwargs.get("masktype", "sigmoid")
        self._maskdict = {'linear': self._linear_interp_mask,
                          'constant': self._constant_region_mask,
                          'sigmoid': self._sigmoid_mask}
        self.mask_kwargs = kwargs.get('mask_kwargs', {})

        self.L = len(self.imag0)
        self._h = kwargs.get("h", 1E-7)
        self._mirror = kwargs.get("mirror", True)

        self.x = np.arange(self.L)
        self.kx = np.fft.rfftfreq(self.L, d=1./(2*np.pi))
        # For mirror padding
        self.kx2 = np.fft.rfftfreq(2*self.L, d=1./(2*np.pi))

    def initialize(self, images):
        self.images = images
        self.imag0, self.imag1 = images
        self.imag1_k = np.fft.rfft(self.imag1)
        self.imag1_mirror = self.mirrorpad(self.imag1)
        self.imag1_mirror_k = np.fft.rfft(self.imag1_mirror)

    def mirrorpad(self, imag):
        return np.concatenate([imag, imag[::-1]])

    def translate(self, delta, imag=None):
        """
        Translate image by delta.
        inputs:
            (required)
            delta : array_like with two elements [dy, dx]
            (optional)
            imag : array_like of shape (Ly, Lx), default is self.imag1
        returns:
            imag_translated : array_like of shape (Ly,Lx) translated by delta
        """
        if self._mirror:
            if imag is None:
                imag_k = self.imag1_mirror_k
            else:
                imag_k = np.fft.rfft(self.mirrorpad(imag))
            kx = self.kx2
        else:
            imag_k = self.imag1_k if imag is None else np.fft.rfft(imag)
            kx = self.kx
        # NOTE: I choose a convention for shifts here
        return np.fft.irfft(imag_k * np.exp(1.j * delta * kx))[:self.L]

    def _linear_interp_mask(self, x, d, **kwargs):
        if d >= 0:
            return np.clip(x+1-d, 0, 1)
        else:
            return np.clip(-x+len(x)+d, 0, 1)

    def _sigmoid_mask(self, x, d, **kwargs):
        w, o = kwargs.get('w', 0.5), kwargs.get('o', 4)
        L = len(x)-1
        if d >= 0:
            return expit((x-d-o)/w)*expit(-(x-L+o)/w)
        else:
            return expit((x-o)/w)*expit(-(x-L-d+o)/w)

    def _constant_region_mask(self, x, d, **kwargs):
        o, l = kwargs.get('o', 0), kwargs.get('l', self.L)
        mask = np.zeros_like(x)
        mask[o:o+l] = 1.
        return mask
    
    def getmask(self, delta):
        mask = self._maskdict[self.masktype]
        return mask(self.x, delta, **self.mask_kwargs)

    def residual(self, delta, images=None):
        """
        Finds the difference translate(imag1) - imag0
        input:
            (required)
            delta : array_like [dy, dx]
            (optional)
            imag1 : array_like image of shape (Ly, Lx) which will be translated
                by delta. Default is self.imag1
            imag0 : array_like iamge of shape (Ly, Lx). Default is self.imag0
        returns:
            residual : array_like, shape can be smaller than (Ly, Lx) if delta
                has components larger in magnitude than 0.5
        """
        if images is not None:
            self.initialize(images)
            
        imag1_delta = self.translate(delta)
        mask = self.getmask(delta)
        res = (imag1_delta - self.imag0)*mask
        return np.nan_to_num(res/np.sqrt(np.sum(mask)))

    def model(self, delta):
        """ Outputs imag1 shifted by delta"""
        imag1_delta = self.translate(delta, self.imag1)
        return imag1_delta * self.getmask(delta)

    def cost(self, delta, images=None):
        """ The summed squared residual. See self.residual """
        r = self.residual(delta, images)
        return r.dot(r)/2.

    def gradres(self, delta, images=None):
        res0 = self.residual(delta, images)
        return (self.residual(delta+self._h, images)-res0)/self._h

    def firstguess(self, images=None):
        """ 
        Naive estimation of the translation by simple cross-correlation
        input:
            (optional)
            imag1 : array_like image of shape (Ly, Lx). Default is self.imag1
            imag0 : array_like iamge of shape (Ly, Lx). Default is self.imag0
        returns:
            delta : array_like of length 2 [dy, dx]
        """
        if images is not None:
            imag0, imag1 = images
        else:
            imag0, imag1 = self.imag0, self.imag1
        L = len(imag1)
        imag1_k, imag0_k = np.fft.rfft(imag1), np.fft.rfft(imag0)
        corr = np.fft.fftshift(np.fft.irfft(imag1_k*imag0_k.conj()))
        maxind = np.argmax(corr)
        return maxind - L/2.

    def estimatenoise(self, delta_bestfit, images=None):
        """
        Estimate the noise at the best fit.
        The log-likelihood is (including normalization)
            ||I0 - T_delta(I1)||^2/(4*sigma^2) + N log sigma
            where N is the number of pixels. Minimizing this w.r.t. sigma
            yields
                sigma = sqrt(C/N) where C = ||I0 - T_delta(I1)||^2
        input:
            delta_bestfit : array_like of length 2 (dy, dx)
            (optional)
            imag1 : array_like image of shape (Ly, Lx). Default is self.imag1
            imag0 : array_like iamge of shape (Ly, Lx). Default is self.imag0
        returns:
            sigma : float, optimal sigma assuming best fit cost
        """
        cost = self.cost(delta_bestfit, images)
        return np.sqrt(2*cost)

    def fit(self, images=None, delta0=None, **kwargs):
        """
        Register two images by optimizing translationg of imag1
        inputs:
            (optional)
            delta0 : array_like of length 2 [dy, dx], default is set by cross
                correlation
            imag1 : array_like image of shape (Ly, Lx). Default is self.imag1
            imag0 : array_like iamge of shape (Ly, Lx). Default is self.imag0
            method : str, 'leastsq' uses leastsq.py, any other options which
                scipy.optimize.minimize accepts
            kwargs:
                are passed directly into leastsq.LM.leastsq or
                scipy.optimize.minimize
        returns:
            p1, p1_sigma, msg : array_like (optimal delta), array_like (error in
                optimal delta), str or int (message from optimizer, see specific
                optimizer for meaning)
        """
        if images is not None:
            self.initialize(images)

        delta0 = delta0 if delta0 is not None else self.firstguess()
        iprint = kwargs.get("iprint", 0)

        if kwargs.get('maskdict'):
            maskdict = kwargs.get('maskdict')
            self.masktype = maskdict.get('masktype', 'linear')
            self.mask_kwargs = maskdict.get('mask_kwargs', [{}, {}])

        #p1 = golden(self.cost, **kwargs)
        self.sol = minimize_scalar(self.cost, bounds=(delta0-2., delta0+2.), 
                                   method='bounded')
        p1 = self.sol['x']
        j = self.gradres(p1)
        jtj = j.dot(j)

        mask = self.getmask(p1)
        sigma = self.estimatenoise(p1)
        p1_sigma = sigma/np.sqrt(jtj*np.sum(mask))
        return p1, p1_sigma
