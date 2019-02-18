"""
periodic_fouriershift.py

author: Colin Clement
date: 2017-06-06

This module computes the rigid-shift registration between two periodic noisy
images, by shifting one image to match the other.
"""

import numpy as np
try:
    from superreg.util.fftw import FFT
    hasfftw = True
except ImportError as ierr:
    print("Install pyfftw for 20x speedup")
    hasfftw = False
import superreg.util.leastsq as leastsq
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

    ASSUMES periodic images
    """
    def __init__(self, images, **kwargs):
        """
        Find the vector [dy, dx] which translates imag1 to match imag0.

        Parameters
        ----------
            (required)
            imag1: array_like image of shape (Ly, Lx)
            imag0 : array_like image of shape (Ly, Lx)
        kwargs:
            h : float, finite difference step size for estimating jacobian
            all other kwargs are passed into FFT (above)

        Usage
        -----
            reg = Register(imag1, imag0)
            delta, delta_sigma, msg = reg.minimize()
        """
        self.images = images
        self.imag0, self.imag1 = images
        assert self.imag0.shape == self.imag1.shape, "Images must share their shape"

        self.Ly, self.Lx = self.imag0.shape
        self.y = np.arange(self.Ly)
        self.x = np.arange(self.Lx)
        self.ky = np.fft.fftfreq(self.Ly, d=1./(2*np.pi))[:,None]
        self.kx = np.fft.fftfreq(self.Lx, d=1./(2*np.pi))

        self.initialize(images)

    def initialize(self, images):
        self.imag0, self.imag1 = images
        n = self.imag0.size
        self.imag0_k = np.fft.fftn(self.imag0, norm='ortho')
        self.imag1_k = np.fft.fftn(self.imag1, norm='ortho')

    def phase(self, delta):
        ky, kx = self.ky, self.kx
        return np.exp(-1.j*delta[0]*ky)*np.exp(-1.j*delta[1]*kx)

    def obj(self, delta, images=None):
        if images is not None:
            self.initialize(images)
        res = self.imag0_k - self.phase(delta)*self.imag1_k
        return np.real(np.sum(res*res.conj()))/2

    def gradobj(self, delta, images=None):
        if images is not None:
            self.initialize(images)
        summand = self.imag0_k*self.phase(-delta)*self.imag1_k.conj()
        ky, kx = self.ky, self.kx
        dky = np.sum(ky*summand)
        dkx = np.sum(kx*summand)
        return np.real(-1j*np.array([dky, dkx]))

    def hessobj(self, delta, images=None):
        if images is not None:
            self.initialize(images)
        summand = self.imag0_k*self.phase(-delta)*self.imag1_k.conj()
        ky, kx = self.ky, self.kx
        d2ky = np.sum(ky*ky*summand).real
        d2kx = np.sum(kx*kx*summand).real  # realfft
        d2kxky = np.sum(kx*ky*summand).real
        return np.array([[d2ky, d2kxky], [d2kxky, d2kx]])

    def fit(self, images=None, delta0=None, **kwargs):
        if images is not None:
            self.initialize(images)
        delta0 = delta0 if delta0 is not None else self.firstguess()
        self.sol = minimize(self.obj, delta0, jac=self.gradobj,
                            hess=self.hessobj, method='Newton-CG', **kwargs)
        sigma = self.estimatenoise(self.sol['x'])
        hess = self.hessobj(self.sol['x'])
        return self.sol['x'], sigma/np.sqrt(hess.diagonal())

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
        n = self.Ly*self.Lx
        if imag is None:
            imag_k = self.imag1_k 
        else:
            imag_k = np.fft.fftn(imag, norm='ortho')
        return np.fft.ifftn(imag_k*self.phase(delta), norm='ortho')

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
        return imag1_delta - imag0

    def res(self, delta, imag1=None, imag0=None):
        """ Helper function which flattens the results of self.residual """
        return self.residual(delta, imag1=None, imag0=None).ravel()

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
        imag1_k, imag0_k = np.fft.fftn(imag1), np.fft.fftn(imag0)
        corr = np.fft.fftshift(np.fft.ifftn(imag1_k*imag0_k.conj()))
        maxind = np.argmax(corr)
        return - np.array([maxind//Lx-Ly/2., maxind % Lx - Lx/2.]) 

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
        cost = self.obj(delta_bestfit, images)
        return np.sqrt(cost/(self.Ly*self.Lx))


if __name__=='__main__':
    import superreg.util.makedata as md
    L = 128
    img = md.powerlaw((L, L), 1.8, scale=L/4)
    d = np.random.randn(2)
    reg = Register(np.random.randn(2,L,L))
    imgshift = reg.translate(-d, img)
    truth = np.array([img, imgshift])
    data = truth + 0.05*np.random.randn(*truth.shape)
    reg = Register(data)
