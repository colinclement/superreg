import os 
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from super_reg.twod.chebseries import SuperRegistration
from super_reg.twod.fouriershift import Register
from util import firsttry
from matt import intshifts


Nmax = 27  # 27 easier dataset

dataloc = "/home/colin/storage/work/data/stem-aberration/bens-registrations"
dataset = "11_unsavedStack_ZoneAxix3.tif"

stack = imread(os.path.join(dataloc, dataset))
stack = stack/float(2**16)
stack = stack[:Nmax]  # complicated images after this
Ly, Lx = stack[0].shape

L = 128 
cutstack = stack[:,:L,:L]

r = Register(stack[:2])
sigma = r.estimatenoise(r.fit()[0])

shift1 = firsttry(cutstack)
#shift2 = firsttry(stack)
#shift1 = firsttry(stack)
#Limit images to shared field of view
#sl = np.prod(1*(shift1 < np.array([L,L])), 1)==1
#ssl = np.s_[9:515,3:508] 
data = intshifts(stack, shift1, L)

#data = cutstack[:sl.cumsum()[-1]-1]
#data = cutstack[:15]

#reg = SuperRegistration(data, 32)
#reg.fit(iprint=1, delta=1E-4, lamb=0.1)

if False:
    #dat = np.load("evidence-opt-2018-08-16.npz")
    dat = np.load("evidence-opt-L-128-2018-08-18.npz")
    deglist = dat['deglist']
    evds = dat['evds']
    shifts = dat['shifts']
    coefs = dat['coefs']
    maxind = np.argmax(evds)
    data = dat['data'] if 'data' in dat else data
    ssl = np.s_[30:580,10:520]
    
    reg = SuperRegistration(data, deglist[maxind], shifts=shifts[maxind][0],
                            coef=coefs[maxind])
    srshifts = reg.shifts + shift1.astype('int')

    #reg = SuperRegistration(data, deglist[maxind], damp=1.)
    #reg.fit(iprint=1, lamb=1E-1)

if True: #__name__=="__main__":

    deglist = range(20, 40)
    
    evds = []
    shifts = []
    coefs = []
    
    for i, d in enumerate(deglist):
        print("deg = {}".format(d))
        s0 = np.random.randn(len(data)-1,2)
        if len(coefs)>0:
            c0 = np.zeros((d+1, d+1))
            c0[:d, :d] = coefs[-1]
        else:
            c0 = None

        reg = SuperRegistration(data, d, shifts=s0, coef=c0)
        shifts.append(reg.fit(iprint=1, delta=1E-4, lamb=0.1))
        evds.append(reg.evidence(sigma=sigma))
        coefs.append(reg.coef)
        print("evd = {}".format(evds[-1]))
    
    today = format(datetime.now()).split()[0]
    
    np.savez("evidence-opt-L-{}-{}".format(L, today), data=data,
             deglist=np.array(list(deglist)), evds=evds, shifts=shifts,
             coefs=coefs, shiftint=shift1.astype('int'))
