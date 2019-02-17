import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as skm
from superreg.fouriershift import Register
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def power(img):
    """
    Compute the power spectrum of an image
    Parameters
    ----------
        img : array_like
            two dimensional image array
    Returns
    -------
        power : array_like
            power spectrum of img, of the same shape as img,
            shifted so that the zero frequency is in the center
    """
    imgk = np.fft.fftn(img)
    imgk[0,0] = 1.
    return np.abs(np.fft.fftshift(imgk))**2

def upsample(img, a=1):
    """
    Given an image and upscaling factor a, use Fourier interpolation to 
    increase the resolution by a factor a.
    input:
    Parameters
    ----------
        img : array_like
            two dimensional image array of shape (Ly, Lx)
        a : int  (default 1)
            upscaling factor, must be positive
    Returns
    -------
        upscaled_img : array_like
            upsampled image of shape (a*Ly, a*Lx)
    """
    assert type(a) is int, "a must be an integer"
    assert a > 0, "a must be positive"
    if a == 1:
        return img
    Ly, Lx = img.shape
    pad = np.zeros((Ly*a, Lx*a), dtype='complex128')
    img_k = np.fft.fftn(img)
    ey = Ly//2 - 1 if not a % 2 else (Ly-1)//2
    ex = Lx//2 - 1 if not a % 2 else (Lx-1)//2
    pad[:ey,:ex] = img_k[:ey, :ex]
    pad[:ey,-ex:] = img_k[:ey, -ex:]
    pad[-ey:,-ex:] = img_k[-ey:, -ex:]
    pad[-ey:,:ex] = img_k[-ey:, :ex]
    return np.fft.ifftn(pad).real * a**2

def coarsen(stack, a):
    """
    Coarsen an image stack with blocks of sidelength a
    Parameters
    ----------
        stack : array_like
            stack of images of shape (N_images, Ly, Lx)
        a : int  (default 1)
            upscaling factor, must be positive
    Returns
    -------
        coarsened_stack : array_like
            stack of images of shape (N_images, Ly//a, Lx//a)
    """
    N, Ly, Lx = stack.shape
    cy, cx = Ly-Ly%a, Lx-Lx%a
    return np.array([skm.block_reduce(s[:cy,:cx], (a,a)) for s in stack])

def multi_image_pairwise_registration(stack, s0list=None, **kwargs):
    """
    Perform multi-image registration via the standard Fourier shift method
    by comparing pairs of adjacent images.
    Parameters
    ----------
        stack : array_like
            stack of images of shape (N_images, Ly, Lx)
        (optional)
        s0list : array_like
            list of initial shift guess of shape (N_images-1, 2)
        kwargs arg handed to superreg.fouriershift.Register
    Returns
    -------
        shifts :  array_like
            shifts of all images with respect to first image, of shape
            (N_images - 1, 2)
    """
    shifts = []
    if s0list is None:
        s0list = [None for i in range(len(stack)-1)]

    for i0, i1, s0 in zip(stack[:-1], stack[1:], s0list):
        reg = Register([i0, i1], **kwargs)
        s, ps = reg.fit(delta0=s0)
        shifts.append(s) 
    return np.array(shifts).cumsum(0)

def coarsereg(stack, a, **kwargs):
    """
    Coarsen before performing registration, then rescale answer
    Parameters
    ----------
        stack : array_like
            stack of images of shape (N_images, Ly, Lx)
        a : int  (default 1)
            upscaling factor, must be positive
    Returns
    -------
        shifts :  array_like
            shifts of all images with respect to first image, of shape
            (N_images - 1, 2)
    """
    return a * multi_image_pairwise_registration(coarsen(stack, a), **kwargs)

def shift_all_to(stack, shifts):
    """
    Given an image stack and shift matrices, find the optimal
    common coordinates, shift each image and average within the
    field of view of the image at the `center' index
    Parameters
    ----------
        stack : array_like
            stack of images of shape (N_images, Ly, Lx)
        shifts : dict 
            with keys 'yij' and 'xij' containing shift matrices
        (optional)
    Returns
    -------
        average_shifted_image : array_like 
            shape (Ly, Lx) of registered and averaged images from stack
    """
    Ly, Lx = stack[0].shape
    oy = int(np.floor(min(min(shifts[:,0]), 0)))
    ox = int(np.floor(min(min(shifts[:,1]), 0)))
    my = int(np.ceil(max(max(shifts[:,0]+Ly), Ly)))
    mx = int(np.ceil(max(max(shifts[:,1]+Lx), Lx)))

    slc = np.s_[abs(oy):abs(oy)+Ly, abs(ox):abs(ox)+Lx]
    model = np.zeros((my-oy, mx-ox))
    avmask = np.zeros_like(model)
    mask = np.zeros_like(model)
    pad = np.zeros_like(model)

    mask[slc] = 1.
    pad[slc] = stack[0].copy()
    reg = Register([pad, pad], masktype='none')
    model += pad
    avmask += mask
    for i, d in enumerate(shifts):
        pad = np.zeros((my-oy, mx-ox))
        pad[slc] = stack[i+1].copy()
        model += reg.translate(d, pad)
        avmask += reg.translate(d, mask)
    model = np.nan_to_num(model/avmask)
    model *= 1*(avmask > 1)
    return model[slc]

def bootstrap(stack, L=64):
    """
    Perform registration on sub-blocks of an image stack to estimate the
    undcertainty in shifts
    Parameters
    ----------
        stack : array_like
            stack of images of shape (N_images, Ly, Lx)
        L : int (default 64)
            size of sub-images to break up the stack into 
            
    Returns
    -------
        shifts : array_like
            list of shift performed over sub-blocks of the image
            of shape (Ly//L * Lx//L, N_images-1, 2)
    """
    Ly, Lx = stack[0].shape
    shifts = []
    for y in range(Ly//L):
        for x in range(Lx//L):
            sl = np.s_[:,y*L:(y+1)*L, x*L:(x+1)*L]
            shifts.append(multi_image_pairwise_registration(stack[sl]))
    return np.array(shifts)

def kompare(stack, shift0, shift1, N, sl=None):
    """
    Compare the results of two different shift estimates in a summary
    plot

    Parameters
    ----------
        stack : array_like
            stack of images of shape (N_images, Ly, Lx)
        shift0 : array_like
            list of shift of shape (N_images-1, 2)
        shift1 : array_like
            list of shift of shape (N_images-1, 2)
        N : int
            only the first N images of the stack will be plotted
        (optional)
        sl : slice_obj
            slice object to crop plotted images
    """
    sl = sl or np.s_[:,:]

    im0 = shift_all_to(stack[:N], shift0[:N-1])
    im1 = shift_all_to(stack[:N], shift1[:N-1])
    im0 = im0[sl]
    im1 = im1[sl]

    def tr(a):
        k = np.fft.fftn(a)
        k[0,0] = 10.0
        k = np.abs(k)**2
        return np.fft.fftshift(k)**0.015

    def trd(a, b):
        ka = np.fft.fftn(a)
        kb = np.fft.fftn(b)
        k = ka - kb
        k[0,0] = 0.0
        k = np.abs(k)**2
        return np.fft.fftshift(k)**0.135

    kim0 = tr(im0)
    kim1 = tr(im1) 
    fig, ax = plt.subplots(1,3, sharex=True, sharey=True)

    vmin = min(kim0.min(), kim1.min())
    vmax = min(kim0.max(), kim1.max())
    ax[0].matshow(kim0, vmin=vmin, vmax=vmax, cmap='Greys')
    ax[0].set_title("Reconstruction from Fourier Shifts")
    ax[1].matshow(kim1, vmin=vmin, vmax=vmax, cmap='Greys')
    ax[1].set_title("Reconstruction from Super Reg Shifts")
    ax[2].matshow(kim1-kim0)
    ax[2].set_title("Difference")
    plt.tight_layout()

def reorder(stack, L):
    """
    Reorders a grid of images so that each image pair in the sequence is a
    nearest neighbor
    """
    for i in range(len(stack)//L):
        if i == 0:
            output = stack[i*L:(i+1)*L]
        elif i % 2 == 0:
            output = np.concatenate([output, stack[i*L:(i+1)*L]])
        else:
            output = np.concatenate([output, stack[i*L:(i+1)*L][::-1]])
    return output

def intshifts(data, shifts, size=64):
    """
    Shift the stack to the nearest integer shift to make plotting easier
    Parameters
    ----------
        stack : array_like
            stack of images of shape (N_images, Ly, Lx)
        shifts : array_like
            list of image shifts of shape (N_images - 1, Ly, Lx)
        size : int (default 64)
            Size of final images        
    Returns
    -------
        cropped_stack : array_like
            shifted and cropped stack of shape (N_images, size, size)
    """
    C = stack.shape[-1]//2 - size//2

    s = shifts.astype('int')
    out = [stack[0][C:C+size, C:C+size]]

    for i in range(len(shifts)):
        d = stack[i+1]
        y, x = s[i]
        out.append(d[C-y:C-y+size, C-x:C-x+size])
    return np.array(out)

def play(stack, interval=1000, **kwargs):
    """
    Play an animation of the sequence of stack images
    Parameters
    ----------
        stack : array_like
            stack of images of shape (N_images, Ly, Lx)
        interval : int
            number of milliseconds between frames
        kwargs are anded to plt.matshow
    Returns
    -------
       fig, ax, animation 
    """
    fig, ax = plt.subplots()
    img = ax.matshow(stack[0], **kwargs)

    def animate(i):
        ax.set_title("Image {}".format(i))
        img.set_data(stack[i])
        return img,

    ani = animation.FuncAnimation(fig, animate, np.arange(len(stack)),
                                  interval=interval)
    plt.show()
    return fig, ax, ani

def corr(i0, i1):
    """
    Compute the cross-correlation between two images
    Parameters
    ----------
        i0 : array_like
        i1 : array_like
            two images of the same shape
    Returns
    -------
        cross_correlation : array_like
    """
    Ly, Lx = (2*np.array(i0.shape)).astype('int')
    i0k = np.fft.rfftn(i0, (Ly, Lx)) 
    i1k = np.fft.rfftn(i1, (Ly, Lx))
    ones = np.fft.rfftn(np.ones_like(i0), (Ly, Lx))
    norm = np.fft.irfftn(ones*ones.conj()).real
    mask = norm==0.
    norm[mask] = 1.
    corr = np.fft.irfftn(i0k*i1k.conj()).real/norm
    corr[mask] = 0.
    return corr
