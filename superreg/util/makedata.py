"""
makedata.py

author: Colin Clement
date: 2017-07-30

This module is for making data to test registration. It takes a large image,
shifts by an arbitrary amount, crops, then adds noise to simulate a registration
problem
"""

import numpy as np
from scipy.misc import face
from scipy.special import expit

data = face().mean(2)/255.
rng = np.random.RandomState(14850)

def mirrorpad(imag):
    """ Double pad imag with mirrors of itself """
    Ly, Lx = imag.shape
    imag_mirror = np.zeros((2*Ly, 2*Lx))
    imag_mirror[:Ly,:Lx] = imag
    imag_mirror[Ly:,:Lx] = imag[::-1]
    imag_mirror[:Ly,Lx:] = imag[:,::-1]
    imag_mirror[Ly:,Lx:] = imag[::-1,::-1]
    return imag_mirror

def sliceL(L, offset=np.array(data.shape)//4):
    """ Return slice object of side-length L """
    oy, ox = map(int, offset)
    return np.s_[oy:oy+L, ox:ox+L]

def gaussianimage(size, y, x, sy, sx):
    """
    Gaussian image
    input: 
        size : tuple of ints, shape of image
        y, x : floats, position of center
        sy, sx: floats, standard deviations
    returns:
        image : array_like of floats
    """
    yy, xx = np.arange(size[0])[:,None], np.arange(size[1])[None,:]
    img = np.exp(-0.5*(((xx-float(x))/sx)**2 + ((yy-float(y))/sy)**2))
    return img/img.sum()

def trianglewaves(size, k=1./3):
    y, x = np.arange(size[0])[:,None], np.arange(size[1])[None,:]
    return (np.sin(-k*x) + np.sin(k*x/2. + np.sqrt(3.)*k*y/2.) +
            np.sin(k*x/2. - np.sqrt(3.)*k*y/2.))

def ellipse(size, y, x, ry, rx, w=0.25):
    yy, xx = np.arange(size[0])[:,None], np.arange(size[1])[None,:]
    r = np.hypot((xx-float(x))/rx, (yy-float(y))/ry)
    return expit((1.-r)*np.sqrt(rx*ry)/w)

def randomblur(size, sx, sy, rng = np.random):
    gauss = gaussianimage(size, size[0]/2, size[1]/2, sy, sx)
    rnd_k = np.fft.fftn(rng.randn(*size))
    img = np.fft.ifftn(rnd_k*np.fft.fftn(np.fft.fftshift(gauss))).real
    return img/img.max()

def powerlaw(size, p, scale=None, rng=np.random):
    ly, lx = size
    scale = scale or min(ly, lx)/2
    kk = rng.randn(ly, lx//2+1) + 1j*rng.randn(ly, lx//2+1)
    x, y = np.arange(lx//2+1), np.fft.fftshift(np.arange(ly)-ly/2)
    xg, yg = np.meshgrid(x, y)
    k = np.hypot(xg, yg)
    img = np.fft.irfftn(kk*np.exp(-(k/scale)**2/2.)/(1+k)**p)
    return img / img.ptp()

def correlatednoise(N, Ly, Lx, scale):
    noise_k = np.fft.fftn(np.random.randn(N, Ly, Lx), axes=(1,2))
    gauss = np.fft.fftshift(gaussianimage((Ly, Lx), Ly/2, Lx/2, scale, scale))
    corrnoise = np.fft.ifftn(noise_k * np.fft.fftn(gauss)[None,...]).real
    corrnoise /= np.sqrt(np.sum(corrnoise * corrnoise))
    return corrnoise*np.sqrt(np.sum(noise_k*noise_k.conj()).real)/np.sqrt(Ly*Lx)

def fakedata(noise, shifts=[np.zeros(2)], L=64, sliceobj=None,
             offset=np.array(data.shape)//4, img=data, mirror=True,
             **kwargs):
    """
    Make fake data from img with relative shifts, 
    some size determined by sliceobj, and some noise added.
    The first image will be unshifted.
    input:
        shifts: list of array_like [y_shift, x_shift]
        sliceobj : tuple, slice objects from sliceL
        noise : float, standard deviation of gaussian noise
        img : ground truth data
    returns:
        images: list of array_like of shape determined by
            sliceobj and img1 shifted w.r.t img0 by delta
    """
    Ly, Lx = img.shape
    sliceobj = sliceobj if sliceobj is not None else sliceL(L, offset)
    if mirror:
        ky = np.fft.fftfreq(2*Ly, d=1./(2*np.pi))[:,None]
        kx = np.fft.rfftfreq(2*Lx, d=1./(2*np.pi))
    else:
        ky = np.fft.fftfreq(Ly, d=1./(2*np.pi))[:,None]
        kx = np.fft.rfftfreq(Lx, d=1./(2*np.pi))
    img0 = img[sliceobj].copy()
    img0 += noise*rng.randn(*img0.shape)
    images = [img0]
    for d in shifts:
        # Shift opposite dir so we solve for shifts with correct sign
        phase = np.exp(1j*d[0]*ky)*np.exp(1j*d[1]*kx)
        if mirror:
            img1 = np.fft.irfftn(np.fft.rfftn(mirrorpad(img))*phase)[:Ly,:Lx]
        else:
            img1 = np.fft.irfftn(np.fft.rfftn(img)*phase)
        img1 = img1[sliceobj]
        img1 += noise*rng.randn(*img1.shape)
        images += [img1]
    return np.array(images)
