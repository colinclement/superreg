import numpy as np
import matplotlib.pyplot as plt
import os

from superreg.util.image import upsample, shift_all_to, power

def diffshifts(reg, fshift, trueshift):
    fig, axe = plt.subplots()
    regdshifts = reg.shifts - trueshift
    fsdshifts = fshift - trueshift
    axe.scatter(regdshifts[:,1], regdshifts[:,0], label="Super Registration")
    axe.scatter(fsdshifts[:,1], fsdshifts[:,0],  label="Fourier Shift")
    axe.legend(loc='best')
    axe.set_title("Inferred Shifts")
    axe.set_xlabel("x-direction (pixels)")
    axe.set_ylabel("y-direction (pixels)")
    axe.set_title("Deviations from true shifts")
    axis = axe.axis()
    axe.set_aspect(np.diff(axis[:2])/np.diff(axis[2:]))
    return fig, axe

def showresults(reg, fshift, data, sigma, trueimg,
                sl=np.s_[1:-2,1:-2], upfactor=2, p=0.5):
    frecon = upsample(shift_all_to(data, fshift)[sl], upfactor)
    ydom, xdom = reg.domain()
    y = np.linspace(0, data.shape[1]-1, upfactor*data.shape[1])
    x = np.linspace(0, data.shape[2]-1, upfactor*data.shape[2])
    xg, yg = np.meshgrid(x, y)
    m = reg(yg, xg)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 9.))
    vmin = min(m.min(), frecon.min())
    vmax = max(m.max(), frecon.max())
    axes[0,0].matshow(m, #+ reg.r.std()*np.random.randn(*m.shape), 
                      vmin=vmin, vmax=vmax, cmap='Greys')
    axes[0,0].axis('off')
    axes[0,0].set_title("Super Registration Model (noise added)")

    axes[0,1].matshow(frecon, vmin=vmin, vmax=vmax, cmap='Greys')
    axes[0,1].axis('off')
    axes[0,1].set_title("Data shifted by Fourier Shifts and averaged")

    axes[0,2].matshow(trueimg, vmin=vmin, vmax=vmax, cmap='Greys')
    axes[0,2].axis('off')
    axes[0,2].set_title("True Image")

    mk, fk = power(m)**p, power(frecon)**p
    tk = power(trueimg)**p
    vmin = min(mk.min(), fk.min())
    vmax = max(mk.max(), fk.max())
    axes[1,0].matshow(mk, vmin=vmin, vmax=vmax, cmap='Greys')
    axes[1,0].axis('off')
    axes[1,0].set_title("Power of SR model")
    axes[1,1].matshow(fk, vmin=vmin, vmax=vmax, cmap='Greys')
    axes[1,1].axis('off')
    axes[1,1].set_title("Power of FS reconstruction")
    axes[1,2].matshow(tk, vmin=vmin, vmax=vmax, cmap='Greys')
    axes[1,2].axis('off')
    axes[1,2].set_title("Power of true image")

    plt.suptitle("Model using {}x{} Chebyshev polynomials, 64 EMPAD images".format(
        reg.deg, reg.deg
    ))
    plt.show()
    return m, frecon
