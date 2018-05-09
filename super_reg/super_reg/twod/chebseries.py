import warnings
import numpy as np
import numexpr as ne
from itertools import chain

from matplotlib.pyplot import *
from numpy.polynomial.chebyshev import chebval, chebval2d
from scipy.linalg import svdvals, solve

from super_reg.util.leastsq import LM
from super_reg.twod.fouriershift import Register
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
            List of 2D coordinates for the first image in the sequence.
            It will be used for the basis of all other coordinates for the other images

        domain : ndarray [2, 2]
            [[ymin, ymax], [xmin, xmax]]

        """
        self.deg = deg
        self.images = images
        self.shifts = shifts
        self.N = len(self.images)
        self.shape = self.images[0].shape

        self.x = 1. * np.arange(self.shape[1])
        self.y = 1. * np.arange(self.shape[0])
        self.yg, self.xg = np.meshgrid(self.y, self.x, indexing='ij')

        if self.shifts is None:
            self.shifts = np.array([self.firstguess(self.images[0], i) for i in
                                    self.images[1:]])

        self.coef = self.estimatecoefs()

    def domain(self, shifts=None):
        """   Smallest rectangle containing all shifted images  """
        shifts = shifts if shifts is not None else self.shifts
        ymin, xmin = np.min(shifts, 0)
        ymax, xmax = np.max(shifts, 0)
        ymin = min(ymin, 0)
        xmin = min(xmin, 0)
        ymax = max(ymax + self.shape[0], self.shape[0]) 
        xmax = max(xmax + self.shape[1], self.shape[1]) 
        return np.array([[ymin, ymax], [xmin, xmax]])

    def set_params(self, params):
        M = 2 * (self.N - 1)
        self.shifts = params[:M].reshape(self.N-1, 2)
        self.coef = params[M:].reshape(* (2 * (self.deg + 1,)))

    @property
    def params(self):
        return np.concatenate((self.shifts.ravel(), self.coef.ravel()))

    @property
    def shiftiter(self):
        return chain([np.zeros(2)], self.shifts)

    @property
    def model(self):
        y, x = self.yg, self.xg
        return np.array([self(y + sy, x + sx) for (sy, sx) in self.shiftiter])

    def coord(self, y, x, shifts=None):
        """ 
        Chebyshev domain is [-1,1]
        """
        dy, dx = self.domain(shifts)
        return (2 * (y - dy[0]) / (dy[1] - dy[0]) - 1,
                2 * (x - dx[0]) / (dx[1] - dx[0]) - 1)

    def __call__(self, y, x, shifts=None):
        cy, cx = self.coord(y, x, shifts)
        return chebval2d(cy, cx, self.coef)

    @property
    def r(self):
        return self.model - self.images

    @property
    def r_k(self):
        rk = np.abs(np.fft.fftn(self.r, axes=(1,2)))**2
        rk[:,0,0] = 0.  # hiding sum
        return np.fft.fftshift(rk, axes=(1,2))

    def res(self, params=None):
        params = params if params is not None else self.params
        self.set_params(params)

        return (self.model - self.images).ravel()

    def gradshifts(self, shifts=None, h=1E-6):
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

    def tmatrix(self, shift):
        sy, sx = shift
        eye = np.identity(self.deg+1)
        d = self.deg + 1
        M = d * d
        T = np.zeros((self.images[0].size, M))
        for m in range(d):
            for n in range(d):
                cy, cx = self.coord(self.y + sy, self.x + sx)
                T[:, m*d+n] = (chebval(cy, eye[m])[:,None] * 
                               chebval(cx, eye[n])[None,:]).ravel()
        return T

    def gradcoef(self):
        M = np.prod(self.shape)
        jac = np.zeros((self.N * M, (self.deg+1)**2))
        eye = np.identity(self.deg+1)
        for i in range(self.deg+1):
            for j in range(self.deg+1):
                for k, (sy, sx) in enumerate(self.shiftiter):
                    cy, cx = self.coord(self.y + sy, self.x + sx)
                    sl = np.s_[k * M:(k + 1) * M, j + i * (self.deg+1)]
                    jac[sl] = (chebval(cy, eye[i])[:,None] *
                               chebval(cx, eye[j])[None,:]).ravel()
        return np.array(jac)

    def grad(self, params=None, h=1E-6):
        if params is not None:
            self.set_params(params)

        gshifts = self.gradshifts(h=h)
        gcoef = self.gradcoef()

        return np.concatenate((gshifts, gcoef), axis=1)

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
            c = p0[i]
            p0[i] = c + h
            res0 = self.res(p0)
            p0[i] = c - h
            res1 = self.res(p0)
            p0[i] = c

            j += [(res0 - res1) / (2*h)]

        return np.array(j).T

    def estimatenoise(self, params=None):
        if params is not None:
            self.set_params(params)
        else:
            params = self.params
        r = self.res(params)
        return np.sqrt(r.dot(r)/len(r))
    
    def firstguess(self, imag1, imag0):
        reg = Register([imag1, imag0])
        return reg.fit()[0]  # convention is opposite for marginal

    def estimatecoefs(self):
        tmats = [self.tmatrix(s) for s in self.shiftiter]
        A = np.sum([t.T.dot(t) for t in tmats], 0)
        b = np.sum([t.T.dot(d.ravel()) for t, d in zip(tmats, self.images)], 0)
        return solve(A, b).reshape(self.deg+1,-1)

    def setoptimizer(self):
        self.opt = LM(self.res, self.grad)

    def fit(self, images=None, p0=None, **kwargs):
        if images is not None:  # reset images and parameters
            self.images = images
        p0 = p0 if p0 is not None else self.params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.setoptimizer()
            self.sol = self.opt.leastsq(p0, **kwargs)
        j = self.grad()
        jtj = j.T.dot(j)
        sigma = self.estimatenoise()
        shifts = self.shifts
        jtjshifts = jtj.diagonal()[:shifts.size].reshape(*shifts.shape)

        return shifts, sigma/np.sqrt(jtjshifts)

    def evidenceparts(self, sigma=None):
        s = sigma if sigma is not None else self.estimatenoise()
        r = self.res()
        N = len(self.params)
        j = self.grad()
        logdet = 2*np.log(svdvals(j)).sum()
        return np.array([-r.dot(r)/s**2, N*np.log(2*np.pi*s**2), -logdet])/2.

    def evidence(self, sigma=None):
        return self.evidenceparts(sigma).sum()


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

    reg = SuperRegistration(data, 18)
    #s1, s1s = reg.fit(iprint=1, delta=1E-8)
    
    deglist = np.arange(6, 26)
    evd = []
    ans = []
    ans_sigma = []
    for d in deglist:
        reg = SuperRegistration(data, d)
        s1, s1s = reg.fit(iprint=0, delta=1E-8, itnlim=100)
        r = reg.res()
            
        evd.append(reg.evidenceparts(sigma=sigma))
        ans.append(s1.squeeze())
        ans_sigma.append(s1s.squeeze())
        print("Finished d={} with evd={}".format(d, evd[-1].sum()))

    evd = np.array(evd)
    ans = np.array(ans)
    ans_sigma = np.array(ans_sigma)
