import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as skm
from super_reg.twod.fouriershift import Register
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def power(img):
    imgk = np.fft.fftn(img)
    imgk[0,0] = 1.
    return np.abs(np.fft.fftshift(imgk))**2

def shiftallto(stack, shifts):
    """
    Given an image stack and shift matrices, find the optimal
    common coordinates, shift each image and average within the
    field of view of the image at the `center' index
    input:
        stack : array_like of shape (N_images, Ly, Lx)
        shifts : dict with keys 'yij' and 'xij' containing
            shift matrices
        (optional)
    returns:
        average_shifted_image : array_like of shape (Ly, Lx)
        of registered and averaged images from stack
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

def upsample(img, a=1):
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

def firsttry(images, s0list=None, **kwargs):
    shifts = []
    if s0list is None:
        s0list = [None for i in range(len(images)-1)]

    for i0, i1, s0 in zip(images[:-1], images[1:], s0list):
        reg = Register([i0, i1], **kwargs)
        s, ps = reg.fit(delta0=s0)
        shifts.append(s) 
    return np.array(shifts).cumsum(0)

def coarsen(stack, a):
    N, Ly, Lx = stack.shape
    cy, cx = Ly-Ly%a, Lx-Lx%a
    return np.array([skm.block_reduce(s[:cy,:cx], (a,a)) for s in stack])

def coarsereg(stack, a, **kwargs):
    return a * firsttry(coarsen(stack, a), **kwargs)

def bootstrap(images, L=64):
    Ly, Lx = images[0].shape
    shifts = []
    for y in range(Ly//L):
        for x in range(Lx//L):
            sl = np.s_[:,y*L:(y+1)*L, x*L:(x+1)*L]
            shifts.append(firsttry(images[sl]))
    return np.array(shifts)

def kompare(data, shift0, shift1, N, sl=None):
    sl = sl or np.s_[:,:]

    im0 = shiftallto(data[:N], shift0[:N-1])
    im1 = shiftallto(data[:N], shift1[:N-1])
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

def reorder(images, L):
    """
    Reorders a grid of images so that each image pair in the sequence is a
    nearest neighbor
    """
    for i in range(len(images)//L):
        if i == 0:
            output = images[i*L:(i+1)*L]
        elif i % 2 == 0:
            output = np.concatenate([output, images[i*L:(i+1)*L]])
        else:
            output = np.concatenate([output, images[i*L:(i+1)*L][::-1]])
    return output

def intshifts(data, shifts, size=64):
    C = data.shape[-1]//2 - size//2

    s = shifts.astype('int')
    out = [data[0][C:C+size, C:C+size]]

    for i in range(len(shifts)):
        d = data[i+1]
        y, x = s[i]
        out.append(d[C-y:C-y+size, C-x:C-x+size])
    return np.array(out)

def play(stack, interval=1000, **kwargs):
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
