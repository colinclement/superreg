import os 
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from super_reg.twod.chebseries import SuperRegistration
from super_reg.twod.fouriershift import Register
from util import firsttry
from matt import intshifts


Nmax = 40  # 27 easier dataset

dataloc = "/home/colin/storage/work/data/stem-aberration/bens-registrations"
dataset = "11_unsavedStack_ZoneAxix3.tif"

stack = imread(os.path.join(dataloc, dataset))
stack = stack/float(2**16)
Ly, Lx = stack[0].shape

L = 64
cutstack = stack[:,:L,:L]

sigma = 0.04  # estimated from Register

shift1 = firsttry(cutstack)
#shift2 = firsttry(stack)
#shift1 = firsttry(stack)
#Limit images to shared field of view
sl = np.prod(1*(shift1 < np.array([L,L])), 1)==1
ssl = np.s_[9:515,3:508]

data = intshifts(stack, shift1, L)

#data = cutstack[:sl.cumsum()[-1]-1]
#data = cutstack[:15]

#reg = SuperRegistration(data, 32)
#reg.fit(iprint=1, delta=1E-4, lamb=0.1)


if True: #__name__=="__main__":

    deglist = range(32, 45)
    
    evds = []
    shifts = []
    coefs = []
    
    for i, d in enumerate(deglist):
        print("deg = {}".format(d))
        s0 = np.random.randn(len(data)-1,2)
        reg = SuperRegistration(data, d, shifts=s0)
        shifts.append(reg.fit(iprint=1, delta=1E-6, lamb=0.1))
        evds.append(reg.evidence(sigma=sigma))
        coefs.append(reg.coef)
        print("evd = {}".format(evds[-1]))
    
    # NOTE that deg=32 is the max evidence
    
    today = format(datetime.now()).split()[0]
    
    np.savez("evidence-opt-{}".format(today),
             deglist=np.array(list(deglist)), evds=evds, shifts=shifts,
             coefs=coefs)
