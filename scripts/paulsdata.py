import os 
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import skimage.measure as skm

from super_reg.twod.chebseries import SuperRegistration
from super_reg.twod.fouriershift import Register

from util import shiftallto, firsttry, upsample

dataloc = "/home/colin/storage/work/data/superreg/SuperRegData"
dataset = ("DUP_44_29Mx_70um_30mrad_ss8_tuned_focused_" + 
           "CL940mm_50x_50y_100z_432step_x128_y128.tif")

stack = imread(os.path.join(dataloc, dataset)).astype('float64')
stack = stack/float(2**16)
minstack = stack.min()
ptpstack = stack.ptp()
stack = (stack - minstack)/ptpstack

# y, x, ky, kx
Ly, Lx, Ky, Kx = stack.shape

bfimage = stack.sum((0,1))
# Center of the BF image
mom = skm.moments(bfimage, order=1)
Cy, Cx = int(mom[1,0]/mom[0,0]), int(mom[0,1]/mom[0,0])

# See the whole disc!
# plt.matshow(stack.sum((0,1)), cmap='Greys_r')
# See the whole bright field image
# plt.matshow(stack[:,:,32:-32,32:-32].sum((2,3)), cmap='Greys_r')

# Do multi image registration on a small set of k points near the center where
# theres little aberration
lky, lkx = 4, 4
bfslfull = np.s_[:,:,Cy-lky:Cy+lky,Cy-lkx:Cx+lkx]

ry, rx = 32, 32
bfsl = np.s_[-ry:,:rx,Cy-lky:Cy+lky,Cy-lkx:Cx+lkx]

cutfull = stack[bfslfull]
cut = stack[bfsl]

data = np.moveaxis(cut.reshape((ry, rx, -1)), -1, 0)
datafull = np.moveaxis(cutfull.reshape((Ly, Lx, -1)), -1, 0)

# Estimate noise
r = Register(datafull[:2])
sigma = r.estimatenoise(r.fit()[0])

fshift = firsttry(data)
fshiftfull = firsttry(datafull)

#reg = SuperRegistration(data, deg=30)
#reg.fit(iprint=1, lamb=0.1)


if False:
    #dat = np.load("paul-evidence-opt-more-2018-08-17.npz")
    #dat = np.load("paul-evidence-opt-even-more-2018-08-18.npz")
    dat = np.load("paul-evidence-opt-even-more-2018-08-20.npz")
    deglist = dat['deglist']
    evds = dat['evds']
    shifts = dat['shifts']
    coefs = dat['coefs']
    maxind = 0 #np.argmax(evds)
    data = dat['data']
    
    reg = SuperRegistration(data, deglist[maxind], shifts=shifts[maxind][0],
                            coef=coefs[maxind])

    #frecon = shiftallto(data, fshiftfull)[:-1,:-1]
    upfactor = 4
    frecon = upsample(shiftallto(data, fshiftfull)[:-1,:-1], upfactor)
    ydom, xdom = reg.domain()
    y = np.linspace(0, 31, upfactor*data.shape[1])
    x = np.linspace(0, 31, upfactor*data.shape[1])
    xg, yg = np.meshgrid(x, y)
    m = reg(yg, xg)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6))
    vmin = data.min()-sigma  # min(m.min(), frecon.min())
    vmax = data.max()+sigma  # max(m.max(), frecon.max())
    axes[0].matshow(m + reg.r.std()*np.random.randn(*m.shape), 
                    vmin=vmin, vmax=vmax, cmap='Greys')
    axes[0].axis('off')
    axes[0].set_title("Super Registration Model (noise added)")
    axes[1].matshow(frecon, vmin=vmin, vmax=vmax, cmap='Greys')
    axes[1].axis('off')
    axes[1].set_title("Data shifted by Fourier Shifts and averaged")
    axes[2].scatter(reg.shifts[:,1], reg.shifts[:,0], label="Super Registration Shifts")
    axes[2].scatter(fshiftfull[:,1], fshiftfull[:,0],  label="Fourier Shift Shifts")
    axes[2].set_title("Inferred Shifts")
    axes[2].set_xlabel("x-direction (pixels)")
    axes[2].set_ylabel("y-direction (pixels)")
    axis = axes[2].axis()
    axes[2].set_aspect(np.diff(axis[:2])/np.diff(axis[2:]))
    plt.suptitle("Model using {}x{} Chebyshev polynomials, 64 EMPAD images".format(
        reg.deg, reg.deg
    ))
    plt.legend()
    plt.show()


if True: #__name__=="__main__":

    deglist = range(79, 90)
    
    evds = []
    shifts = []
    coefs = []
    
    for i, d in enumerate(deglist):
        print("deg = {}".format(d))
        reg = SuperRegistration(data, d)
        shifts.append(reg.fit(iprint=2, delta=1E-2, lamb=0.1))
        evds.append(reg.evidence(sigma=sigma))
        coefs.append(reg.coef)
        print("evd = {}".format(evds[-1]))
    
    today = format(datetime.now()).split()[0]
    
    np.savez("paul-evidence-opt-{}".format(today),
             deglist=np.array(list(deglist)), evds=evds, shifts=shifts,
             coefs=coefs, data=data)
