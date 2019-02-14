"""
fouriershift.py

author: Colin Clement
date: 2017-06-06

This module computes the registration error between two images, i.e. it finds a shift
which most matches two images by finding an optimal shift to match the two.
"""

import numpy as np
try:
    from super_reg.util.fftw import FFT
    hasfftw = True
except ImportError as ierr:
    print("Install pyfftw for 20x speedup")
    hasfftw = False
import super_reg.util.leastsq as leastsq
from scipy.optimize import minimize
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
            delta, delta_sigma, msg = reg.minimize()
        """
        self.images = images
        self.imag0, self.imag1 = images
        self.Ly, self.Lx = self.imag0.shape
        assert self.imag0.shape == self.imag1.shape, "Images must share their shape"
        self.masktype = kwargs.get("masktype", "constant")
        self._maskdict = {'linear': self._linear_interp_mask,
                          'constant': self._constant_region_mask,
                          'sigmoid': self._sigmoid_mask,
                          'none': self._nomask}
        defaultkwargs = [{'o': 4, 'l': self.Ly-8}, {'o': 4, 'l': self.Lx-8}]
        self.mask_kwargs = kwargs.get('mask_kwargs', defaultkwargs)

        self._h = kwargs.get("h", 1E-7)
        self._mirror = kwargs.get("mirror", True)
        self._dy = np.array([self._h, 0.])
        self._dx = np.array([0., self._h])

        self.y = np.arange(self.Ly)
        self.x = np.arange(self.Lx)
        self.ky = np.fft.fftfreq(self.Ly, d=1./(2*np.pi))[:,None]
        self.kx = np.fft.rfftfreq(self.Lx, d=1./(2*np.pi))
        # For mirror padding
        self.ky2 = np.fft.fftfreq(2*self.Ly, d=1./(2*np.pi))[:,None]
        self.kx2 = np.fft.rfftfreq(2*self.Lx, d=1./(2*np.pi))

        if hasfftw:
            self.FFT = FFT(self.imag0.shape, **kwargs)
            self.FFT2 = FFT(2*np.array(self.imag0.shape), **kwargs)
            self.fft2 = self.FFT.fft2
            self.ifft2 = self.FFT.ifft2
            self.fft22 = self.FFT2.fft2
            self.ifft22 = self.FFT2.ifft2
        else:
            self.fft2 = np.fft.rfft2
            self.ifft2 = np.fft.irfft2
            self.fft22 = np.fft.rfft2
            self.ifft22 = np.fft.irfft2
        
        self.initialize(images)

    def initialize(self, images):
        self.imag1_k = self.fft2(self.imag1)
        self.imag1_mirror = self.mirrorpad(self.imag1)
        self.imag1_mirror_k = self.fft22(self.imag1_mirror)

    def mirrorpad(self, imag):
        Ly, Lx = imag.shape
        imag_mirror = np.zeros((2*Ly, 2*Lx))
        imag_mirror[:Ly,:Lx] = imag
        imag_mirror[Ly:,:Lx] = imag[::-1]
        imag_mirror[:Ly,Lx:] = imag[:,::-1]
        imag_mirror[Ly:,Lx:] = imag[::-1,::-1]
        return imag_mirror

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
                imag_k = self.fft22(self.mirrorpad(imag))
            ky, kx = self.ky2, self.kx2
            ifft2 = self.ifft22
        else:
            imag_k = self.imag1_k if imag is None else self.fft2(imag)
            ky, kx = self.ky, self.kx
            ifft2 = self.ifft2
        phase = np.exp(-1.j*delta[0]*ky)*np.exp(-1.j*delta[1]*kx)
        return ifft2(imag_k*phase)[:self.Ly, :self.Lx]

    def _linear_interp_mask(self, x, d, **kwargs):
        if d >= 0:
            return np.clip(x+1-d, 0, 1)
        else:
            return np.clip(-x+len(x)+d, 0, 1)

    def _sigmoid_mask(self, x, d, **kwargs):
        w, o = kwargs.get('w', 0.25), kwargs.get('o', 2)
        L = len(x)-1
        if d >= 0:
            return expit((x-d-o)/w)*expit(-(x-L+o)/w)
        else:
            return expit((x-o)/w)*expit(-(x-L-d+o)/w)

    def _constant_region_mask(self, x, d, **kwargs):
        o, l = kwargs.get('o', 0), kwargs.get('l', self.Lx)
        mask = np.zeros_like(x)
        mask[o:o+l] = 1.
        return mask

    def _nomask(self, x, d, **kwargs):
        return np.ones_like(x)
    
    def getmask(self, delta, **kwargs):
        mask = self._maskdict[self.masktype]
        ym = mask(self.y, delta[0], **self.mask_kwargs[0])
        xm = mask(self.x, delta[1], **self.mask_kwargs[1])
        return np.outer(ym, xm)

    def residual(self, delta, imag1=None, imag0=None):
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
        imag1_delta = self.translate(delta, imag1)
        imag0 = imag0 if imag0 is not None else self.imag0
        mask = self.getmask(delta)
        res = (imag1_delta - imag0)*mask
        return res/np.sqrt(np.sum(mask))

    def model(self, delta, imag1=None):
        """ Outputs imag1 shifted by delta"""
        imag1_delta = self.translate(delta, imag1)
        mask = self.getmask(delta)
        return imag1_delta * mask

    def res(self, delta, imag1=None, imag0=None):
        """ Helper function which flattens the results of self.residual """
        return self.residual(delta, imag1=None, imag0=None).ravel()

    def cost(self, delta, imag1=None, imag0=None):
        """ The summed squared residual. See self.residual """
        r = self.res(delta, imag1, imag0)
        return r.dot(r)/2.

    def gradres(self, delta, imag1=None, imag0=None):
        """ The jacobian of the residual. See self.residual """
        res0 = self.res(delta, imag1, imag0) 
        delta_dy = delta + self._dy
        delta_dx = delta + self._dx
        res_dy = self.res(delta_dy, imag1, imag0) - res0
        res_dx = self.res(delta_dx, imag1, imag0) - res0
        return np.concatenate([[res_dy], [res_dx]]).T/self._h

    def gradcost(self, delta, imag1=None, imag0=None):
        """ The derivative of the cost function. See self.residual """
        c0 = self.cost(delta, imag1, imag0)
        delta_dy = delta + self._dy
        delta_dx = delta + self._dx
        c_dy = self.cost(delta_dy, imag1, imag0) - c0
        c_dx = self.cost(delta_dx, imag1, imag0) - c0
        return np.array([c_dy, c_dx])/self._h

    def firstguess(self, imag1=None, imag0=None):
        """ 
        Naive estimation of the translation by simple cross-correlation
        input:
            (optional)
            imag1 : array_like image of shape (Ly, Lx). Default is self.imag1
            imag0 : array_like iamge of shape (Ly, Lx). Default is self.imag0
        returns:
            delta : array_like of length 2 [dy, dx]
        """
        imag1 = imag1 if imag1 is not None else self.imag1
        imag0 = imag0 if imag0 is not None else self.imag0
        assert imag1.shape==imag0.shape, "Images must share shapes"
        Ly, Lx = imag1.shape
        imag1_k, imag0_k = self.fft2(imag1), self.fft2(imag0)
        corr = np.fft.fftshift(self.ifft2(imag1_k*imag0_k.conj()))
        maxind = np.argmax(corr)
        return - np.array([maxind//Lx-Ly/2., maxind % Lx - Lx/2.]) 

    def estimatenoise(self, delta_bestfit, imag1=None, imag0=None):
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
        cost = self.cost(delta_bestfit, imag1, imag0)
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
            self.images = images
            self.imag0, self.imag1 = images
            self.initialize(images)
        
        delta0 = delta0 if delta0 is not None else self.firstguess()
        iprint = kwargs.get("iprint", 0)

        if kwargs.get('maskdict'):
            maskdict = kwargs.get('maskdict')
            self.masktype = maskdict.get('masktype', 'linear')
            self.mask_kwargs = maskdict.get('mask_kwargs', [{}, {}])

        method = kwargs.get('method', 'Nelder-Mead')
        if method == "leastsq":
            if iprint:
                print("Using method {}".format(method))
            lm = leastsq.LM(self.res, self.gradres)
            p1, jtjdiag, msg = lm.leastsq(delta0, **kwargs)
        else:
            if iprint:
                print("Using method {}".format(method))
            if "iprint" in kwargs:
                iprint = kwargs.pop("iprint")
            if not method in ["Nelder-Mead", "Powell"]:
                jac = self.gradcost
            else:
                jac = None
            sol = minimize(self.cost, delta0, method=method, jac=jac,
                           options={'disp': iprint}, **kwargs)
            J = self.gradres(sol['x'])
            JTJ = J.T.dot(J)
            p1, jtjdiag = sol['x'], JTJ.diagonal(),
            msg = sol['message']

        mask = self.getmask(p1)
        sigma = self.estimatenoise(p1)
        p1_sigma = sigma/np.sqrt(jtjdiag*np.sum(mask))
        return p1, p1_sigma
