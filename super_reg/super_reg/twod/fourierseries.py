import warnings
import numpy as np
import numexpr as ne

from scipy.optimize import golden, leastsq
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter

from super_reg.util.leastsq import LM

DEGREE = 20

rng = np.random.RandomState(14850)

#TODO: 
#   1. Test for biases with finite difference derivative
#   2. Implement correct gradient for coefficients
#   3. Make and test fake data that isn't perfectly representable
#   4. Investigate model complexity


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

        if self.shifts is None:
            self.shifts = rng.rand(self.N-1, 2)

        self.coef = np.random.randn(deg+1, deg+1)

        self.x = 1. * np.arange(self.shape[1])
        self.y = 1. * np.arange(self.shape[0])
        #self.yx = np.meshgrid(self.y, self.x, indexing='ij')
        #self.yx = np.rollaxis(np.array(self.yx), 0, 3)

        self.ky = 2 * np.pi * np.arange(self.deg+1)
        self.kx = 2 * np.pi * np.arange(self.deg+1)
        #self.ky = 2 * np.pi * np.arange(-self.deg, self.deg+1)
        #self.kx = 2 * np.pi * np.arange(-self.deg, self.deg+1)
        #self.kyx = 2 * np.pi * self.yx

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
    def model(self):
        y, x = self.y, self.x
        return np.array([self(y, x)] + 
                        [self(y + sy, x + sx) for (sy, sx) in self.shifts])

    @property
    def c(self):
        return np.concatenate((self.coef[:,1:][:,::-1], self.coef), axis=1)

    def coord(self, y, x, shifts=None):
        """ 
        Note that we use a quarter of domain as we impose mirror symmetry
        """
        dy, dx = self.domain(shifts)
        return ((y - dy[0]) / (2. * (dy[1] - dy[0])),
                (x - dx[0]) / (2. * (dx[1] - dx[0])))

    def __call__(self, y, x, shifts=None):
        cy, cx = self.coord(y, x, shifts)
        cy, cx = cy[None,:], cx[None,:]
        ky, kx = self.ky[:, None], self.kx[:, None]
        #xarg = ne.evaluate('exp(-1j * kx * cx)')
        #yarg = ne.evaluate('exp(-1j * ky * cy)')
        # NOTE: This ensures mirror symmetry
        xarg = ne.evaluate('cos(kx * cx)')
        yarg = ne.evaluate('cos(ky * cy)')
        return self.coef.T.dot(yarg).T.dot(xarg)

    @property
    def residual(self):
        return self.model - self.images

    @property
    def residual_k(self):
        return np.abs(np.fft.fftn(self.residual, axes=(1,2)))**2

    def res(self, params=None):
        params = params if params is not None else self.params
        self.set_params(params)

        return (self.model - self.images).ravel()

    #def gradshifts(self, shifts=None):
    #    shifts = shifts if shifts is not None else self.shifts

    #    args = self.coord(shifts[:,None] * self.k[None,:])
    #    sinkd = ne.evaluate('sin(args)')
    #    coskd = ne.evaluate('cos(args)')

    #    cn = self.An*self.k
    #    cm = self.Bn*self.k

    #    dIds_n = (cn*coskd).dot(self.coskx) - (cn*sinkd).dot(self.sinkx)
    #    dIds_m = (cm*sinkd).dot(self.coskx) + (cm*coskd).dot(self.sinkx)

    #    return (dIds_n - dIds_m) / np.diff(self.domain)
    def gradshifts(self, shifts=None):
        shifts = shifts if shifts is not None else self.shifts



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
            #lm = LM(self.res, self.grad)
            lm = LM(self.res, self.jac)
            self.sol = lm.leastsq(p0, **kwargs)
        #j = self.grad()
        j = self.jac()
        jtj = j.T.dot(j)
        sigma = self.estimatenoise()
        shifts = self.shifts
        jtjshifts = jtj.diagonal()[:shifts.size].reshape(*shifts.shape)

        return shifts, sigma/np.sqrt(jtjshifts)


if __name__=="__main__":
    
    deg = 8
    datamaker = SuperRegistration(np.zeros((2,32,32)), deg=deg)
    images = datamaker.model
    shifts = datamaker.shifts.copy()
    images /= images.std()
    data = images + 0.05 * np.random.randn(*images.shape)

    reg = SuperRegistration(data, deg)
    s1, s1_sigma = reg.fit(iprint=2,)

