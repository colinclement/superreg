"""
chebseries.py

author: Colin Clement and Matt Bierbaum
date: 2018-01-02

This module computes the rigid shift registration between two (or more!)
shifted noisy images, by modeling the true underlying image as sum of
Chebyshev polynomials.

NOTE: As the shifts are updated, the domain of the Chebyshev polynomials is
modified to be the smallest rectangle containing all the images, so that when
the outermost shifts change, the definition of the Chebyshev coeffcients change
too!
"""
import warnings
import numpy as np
import numexpr as ne
from itertools import chain

import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebval, chebval2d
from scipy.linalg import svdvals, solve, lstsq
from scipy.sparse.linalg import cg, lsmr

from superreg.util.leastsq import LM
from superreg.fouriershift import Register


class SuperRegistration(object):
    def __init__(self, images, deg, shifts=None, coef=None, domain=None,
                 **kwargs):
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

        Usage
        -----
        reg = SuperRegistration(image_stack, 10)
        reg.fit(iprint=1)  # will print stages of optimization
        """
        self.deg = deg
        self.images = images
        self.shifts = shifts
        if shifts is not None:
            if not shifts.shape == (len(images)-1, 2):
                raise ValueError("shifts is the incorrect shape")

        self.N = len(self.images)
        self.shape = self.images[0].shape

        self.x = 1. * np.arange(self.shape[1])
        self.y = 1. * np.arange(self.shape[0])
        self.yg, self.xg = np.meshgrid(self.y, self.x, indexing='ij')

        self.firststep(shifts, coef, damp=kwargs.get('damp', 1E-5))

    def firststep(self, shifts=None, coef=None, **kwargs):
        """
        Compute the initial condition using the standard Fourier shift
        registration and a linear fit of the chebyshev coefficients
        Parameters
        ----------
        shifts : array_like 
            use these initial shifts
        coef : array_like
            use these initial Chebyshev coefficients
        """
        self.shifts = shifts
        if shifts is None:
            self.shifts = np.array([
                self.firstguess(i0, i1) for i0, i1 in zip(self.images[:-1], 
                                                          self.images[1:])
            ]).cumsum(0)
        self.coef = coef
        if coef is None:
            self.coef = self.bestcoef(**kwargs)
        return np.concatenate([self.shifts.ravel(), self.coef.ravel()])

    def domain(self, shifts=None):
        """   Smallest rectangle containing all shifted images  """
        shifts = shifts if shifts is not None else self.shifts
        ymin, xmin = np.min(shifts, 0)
        ymax, xmax = np.max(shifts, 0)
        ymin = min(ymin, 0.)
        xmin = min(xmin, 0.)
        ymax = max(ymax + self.shape[0], self.shape[0]) - 1.
        xmax = max(xmax + self.shape[1], self.shape[1]) - 1.
        return np.array([[ymin, ymax], [xmin, xmax]])

    def set_params(self, params):
        """ Take params array and populate shifts and coefs """
        M = 2 * (self.N - 1)
        self.shifts = params[:M].reshape(self.N-1, 2)
        self.coef = params[M:].reshape(* (2 * (self.deg + 1,)))

    @property
    def params(self):
        """ Return the shifts, coefs in one array """
        return np.concatenate((self.shifts.ravel(), self.coef.ravel()))

    @property
    def shiftiter(self):
        """ 
        Add zero shift to shifts to produce a shift iterator the length of
        the number of images
        """
        return chain([np.zeros(2)], self.shifts)

    @property
    def model(self):
        """
        Return the model images in the currrent state of shifts and coefs
        """
        y, x = self.yg, self.xg
        return np.array([self(y + sy, x + sx) for (sy, sx) in self.shiftiter])

    def coord(self, y, x, shifts=None):
        """ 
        Map y, x into the Chebyshev domain of [-1,1]
        """
        dy, dx = self.domain(shifts)
        return (2 * (y - dy[0]) / (dy[1] - dy[0]) - 1,
                2 * (x - dx[0]) / (dx[1] - dx[0]) - 1)

    def gradcoord(self, c, s, ds, yorx, shifts=None):
        """ 
        Compute the gradient of the log-likelihood with respect to the
        coordinate change
        """
        shifts = shifts if shifts is not None else self.shifts
        ss = shifts[:,yorx]
        minshifts, maxshifts = np.min(ss), np.max(ss)
        L = self.images.shape[yorx+1] - 1
        dom = self.domain(shifts)[yorx]
        D = np.diff(dom)[0]
        c0p = 1. if ds == minshifts and ds <= 0 else 0.
        c1p = 1. if ds == maxshifts and ds >= 0 else 0.

        dcoord = 2 * (((s==ds)-c0p)/D - (c + s - dom[0])*(c1p - c0p)/D**2)
        return dcoord

    def __call__(self, y, x, shifts=None):
        cy, cx = self.coord(y, x, shifts)
        return chebval2d(cy, cx, self.coef)

    @property
    def r(self):
        """ The residual error between the model and data """
        return self.model - self.images

    @property
    def r_k(self):
        """ The Fourier spectrum of the residuals"""
        rk = np.abs(np.fft.fftn(self.r, axes=(1,2)))**2
        rk[:,0,0] = 0.  # hiding sum
        return np.fft.fftshift(rk, axes=(1,2))

    def res(self, params=None):
        """ Model residuals flattened into an array """
        params = params if params is not None else self.params
        self.set_params(params)

        return (self.model - self.images).ravel()

    def gradshifts(self, shifts=None, jac=None):
        """ Gradient of log-likelihood with respect to shifts """
        N = np.prod(self.shape)
        shifts = shifts if shifts is not None else self.shifts
        minshifts, maxshifts = np.min(shifts, 0), np.max(shifts, 0)

        if jac is None:
            jac = np.zeros((self.N * np.prod(self.shape), self.shifts.size))

        # loop over images in residuals
        for i, si in enumerate(chain([np.zeros(2)], shifts)):
            # loop over shift parameters
            for j, sj in enumerate(shifts.flat): 
                mins = sj == minshifts[j%2] and sj <= 0
                maxs = sj == maxshifts[j%2] and sj >= 0
                if mins or maxs or j//2 == i-1:
                    gradt = self._tmatrix(si, grady=not j%2, gradx=j%2)
                    ckl = self.coef[:,1:].ravel() if j%2 else self.coef[1:].ravel()

                    c = self.x[None,:] if j%2 else self.y[:,None]
                    gradc = self.gradcoord(c, si[j%2], sj, j%2, shifts)

                    jac[i*N:(i+1)*N, j] = np.ravel(
                        gradt.dot(ckl).reshape(*self.shape)*gradc
                    )
        return jac

    def gradshifts_fd(self, shifts=None, h=1E-6):
        """ Finite difference gradient for debugging """
        if shifts is not None:
            self.shifts = shifts

        jac = np.zeros((self.N * np.prod(self.shape), self.shifts.size))
        p0 = self.params.copy()
        for i in range(self.shifts.size):
            c = p0[i]
            p0[i] = c + h
            r0 = self.res(params=p0)
            p0[i] = c - h
            r1 = self.res(params=p0)
            p0[i] = c
            jac[:,i] = (r0 - r1)/(2 * h)
        return jac

    def gradcoef(self, jac=None):
        """ Gradient of the log-likelihood with respect to coefs """
        M = np.prod(self.shape)
        if jac is None:
            jac = np.zeros((self.N * M, (self.deg+1)**2))
        for k, s in enumerate(self.shiftiter):
            jac[k*M:(k+1)*M,:] = self._tmatrix(s)
        return jac

    def grad(self, params=None):
        """ The total gradient of the log-likelihood """
        if params is not None:
            self.set_params(params)
        if hasattr(self, '_jac'):
            self._jac.fill(0.)
        else:
            self._jac = np.zeros((self.images.size, self.params.size))
        Ns = self.shifts.size
        self.gradshifts(jac=self._jac[:,:Ns])
        self.gradcoef(jac=self._jac[:,Ns:])
        return self._jac

    def jac(self, p=None, h=1e-6):
        """ The finite-different gradient for debugging """
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
            res0 = self.res(p0)
            p0[i] = c - h
            res1 = self.res(p0)
            p0[i] = c

            j += [(res0 - res1) / (2*h)]

        return np.array(j).T

    def estimatenoise(self, params=None):
        """
        Estimate the noise of the data. Should be at best fit parameters.
        """
        if params is not None:
            self.set_params(params)
        else:
            params = self.params
        r = self.res(params)
        return np.sqrt(0.5*r.dot(r)/(len(r)-len(params)))
    
    def firstguess(self, imag1, imag0):
        """ Wrapper for Fouriershift registration """
        reg = Register([imag1, imag0])
        return reg.fit()[0]  # convention is opposite for marginal

    def _tmatrix(self, shift, grady=False, gradx=False):
        """
        Compute the matrix which computes all Chebyshev images from a vector of
        coefficients, i.e. Images = T.dot(coefs)
        """
        dy = self.deg + 1 if not grady else self.deg
        dx = self.deg + 1 if not gradx else self.deg
        T = np.zeros((self.images[0].size, dx*dy))
        cy, cx = self.coord(self.y + shift[0], self.x + shift[1])
        Ty0 = np.ones_like(cy)
        Ty1 = cy if not grady else 2 * cy
        for m in range(dy):
            Tx0 = np.ones_like(cx)
            Tx1 = cx if not gradx else 2 * cx
            for n in range(dx):
                if grady and gradx:
                    T[:, m*dx+n] = ((m+1) * Ty0[:,None] * (n+1) * Tx0[None,:]).ravel()
                elif grady:
                    T[:, m*dx+n] = ((m+1) * Ty0[:,None] * Tx0[None,:]).ravel()
                elif gradx:
                    T[:, m*dx+n] = (Ty0[:,None] * (n+1) * Tx0[None,:]).ravel()
                else:
                    T[:, m*dx+n] = (Ty0[:,None] * Tx0[None,:]).ravel()
                Txn = Tx1
                Tx1, Tx0 = 2 * cx * Tx1 - Tx0, Txn
            Tyn = Ty1
            Ty1, Ty0 = 2 * cy * Ty1 - Ty0, Tyn
        return T

    def bestcoef(self, **kwargs):
        """ With fixed shifts find best fit coefficients """
        M = (self.deg + 1)**2
        A = np.zeros((M, M))
        b = np.zeros(M)
        for s, d in zip(self.shiftiter, self.images):
            t = self._tmatrix(s)
            A[:,:] += t.T.dot(t)
            b[:] += t.T.dot(d.ravel())

        self._firstcoef = lsmr(A, b, **kwargs)
        return self._firstcoef[0].reshape(self.deg+1,-1)

    def bestshifts(self, shifts=None, **kwargs):
        """ With fixed coefs find best fit shifts """
        if shifts is not None:
            self.shifts = shifts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = lambda s: self.res(np.concatenate([s, self.coef.ravel()]))
            grad = lambda s: self.gradshifts(s.reshape(self.shifts.shape))
            opt = LM(res, grad)
            sol = opt.leastsq(self.shifts.ravel(), **kwargs)
        return sol[0].reshape(self.shifts.shape)

    def setoptimizer(self):
        self.opt = LM(self.res, self.grad)

    def paramerrors(self, params=None, sigma=None):
        """ Compute CRB of parameters at best fit """
        sigma = sigma if sigma is not None else self.estimatenoise()
        j = self.grad(params)
        jtj = j.T.dot(j)
        return sigma/np.sqrt(jtj.diagonal())

    def fit(self, images=None, p0=None, sigma=None, **kwargs):
        """
        Maximize the log-likelihood
        Can set or reset the data (images), initial parameters,
        the noise in the data (sigma).
        Parameters
        ----------
        (optional)
        images : array_like
            stack of data images of shape (N_images, Ly, Lx)
        p0 : array_like
            flat array of image parameters (shifts, coefs)
        sigma : float
            estimated noise in images
        kwargs:
            handed to the optimizer, superreg.util.leastsq.LM by default
            iprint = 0, 1, 2 sets the verbosity of the optimizer. See
            documentation of LM for more kwargs.
        """
        if images is not None:  # reset images and parameters
            self.images = images
        p0 = p0 if p0 is not None else self.params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.setoptimizer()
            self.sol = self.opt.leastsq(p0, **kwargs)
        perrors = self.paramerrors(sigma=sigma)
        shifts = self.shifts

        return shifts, perrors[:shifts.size].reshape(*shifts.shape)

    def cost(self, params=None):
        """ Log-likelihood of data given parameters """
        return np.sum(self.res(params)**2)/2.
    
    def itnfit(self, images=None, p0=None, sigma=None, tol=1E-6, **kwargs):
        """
        Alternative optimization scheme of alternating minimization of shifts
        and coefficients.
        """
        if images is not None:  # reset images and parameters
            self.images = images
        if p0 is not None:
            self.set_params(p0)
        
        r0 = self.cost()
        converged = False
        while not converged:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.coef = self.bestcoef()
                self.shifts = self.bestshifts(**kwargs)
            r1 = self.cost()
            f = np.abs((r1 - r0)/r0)
            if kwargs.get('iprint', 0)+1>0:
                print("{} {} ".format(r1, f))
            if f < tol:
                converged = True
            r0 = r1

        perrors = self.paramerrors(sigma=sigma)
        shifts = self.shifts

        return shifts, perrors[:shifts.size].reshape(*shifts.shape)

    def evidenceparts(self, sigma=None):
        s = sigma if sigma is not None else self.estimatenoise()
        r = self.res()
        N = len(self.params)
        j = self.grad()
        logdet = 2*np.log(svdvals(j)).sum()
        return np.array([-r.dot(r)/s**2, N*np.log(2*np.pi*s**2), -logdet])/2.

    def evidence(self, sigma=None):
        """ Compute the model evidence or the normalization of the posterior """
        return self.evidenceparts(sigma).sum()

    def show(self, n=2, cmap='Greys_r'):
        """ 
        Plot n of the model images and their residuals in real and Fourier
        space 
        """
        fig, axes = plt.subplots(4, n)
        kw = {'vmin' : min(self.images.min(), self.model.min()),
              'vmax' : max(self.images.max(), self.model.max()),
              'cmap': cmap}
        for i in range(n):
            axes[0,i].matshow(self.images[i], **kw)
            axes[1,i].matshow(self.model[i], **kw)
            axes[2,i].matshow(self.r[i])
            axes[3,i].matshow(self.r_k[i], cmap=cmap)
        for a in axes.flat:
            a.axis('off')
        plt.show()


def optcomplexity(data, sigma,  **kwargs):
    """
    A simple scheme for evidence maximization which increases complexity
    until the evidence decreases to find the most likely model
    """
    d0 = kwargs.pop('d0', 5)
    show = kwargs.pop('show', False)
    reg = SuperRegistration(data, d0)
    reg.fit(sigma=sigma, **kwargs)
    e0 = reg.evidence(sigma)
    converged = False
    evdlist = []
    while not converged:
        if show:
            print("d={}, evd={}".format(d0, e0))
        reg = SuperRegistration(data, d0+1)
        e1 = reg.evidence(sigma)
        if e1 < e0:  # if evidence decreases with increasing complexity
            converged = True
        else:
            d0 += 1
            e0 = e1
    return d0
 
if __name__=="__main__":
    
    import matplotlib.pyplot as plt
    import superreg.util.makedata as md

    deg = 8
    L = 32 rng = np.random.RandomState(14850) 
    img = md.powerlaw((2*L, 2*L), 1.8, scale=2*L/6., rng=rng)
    shifts = [np.array([.15, 0.])]  # rng.randn(2)
    shifts = np.random.randn(10,2)
    images = md.fakedata(0., shifts, L, img=img, offset=L*np.ones(2),
                         mirror=False)
    images /= images.ptp()

    sigma = 0.05
    data = images + sigma * rng.randn(*images.shape)

    reg = SuperRegistration(data, 16)
    p = reg.params.copy()
    #s1, s1s = reg.fit(p0=p, iprint=0, delta=1E-8)
    #s1i, s1si = reg.itnfit(p0=p, iprint=0, tol=1E-8, delta=1E-8)
    
    deglist = np.arange(6, 26)
    evd = []
    ans = []
    ans_sigma = []
    #for d in deglist:
    #    reg = SuperRegistration(data, d)
    #    s1, s1s = reg.fit(iprint=0, delta=1E-8, itnlim=100)
    #    r = reg.res()
    #        
    #    evd.append(reg.evidenceparts(sigma=sigma))
    #    ans.append(s1.squeeze())
    #    ans_sigma.append(s1s.squeeze())
    #    print("Finished d={} with evd={}".format(d, evd[-1].sum()))

    evd = np.array(evd)
    ans = np.array(ans)
    ans_sigma = np.array(ans_sigma)
