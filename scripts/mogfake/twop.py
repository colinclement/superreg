import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from superreg.chebseries import SuperRegistration
from superreg.fouriershift import Register
import makedata as md

os.sys.path.append('../')
import util
from mog import mog

mpl.rcParams['image.cmap'] = 'Greys'

rng = np.random.RandomState(92089)

def makemog(data, pos, widths, sigma):
    Ly, Lx = data.shape
    m = mog.Mog(data, boxprior=0., anisprior=0.)
    p0 = [Ly, Ly*pos[0][1], -Ly*pos[0][0], Ly*widths[0],
          Ly, Ly*pos[1][1], -Ly*pos[1][0], Ly*widths[1],
          0., 1., sigma, 0.]
    pnames = []
    pnames += [n.format(0) for n in m.localnameformat]
    pnames += [n.format(1) for n in m.localnameformat]
    pnames += m.globalnames
    m.register(pnames, p0)
    return m

N = 8
L = 64
sigma = 0.2
s0 = rng.randn(N-1, 2)/(L/2)
shifts = L * s0

pos = [0.3*rng.rand(2)+0.0,
       0.4*rng.rand(2)+0.5]
widths = [0.1, 0.16]
model = md.twoparticles(np.concatenate([[np.zeros(2)], s0]), pos=pos,
                        widths=widths)
data = model + sigma * rng.randn(*model.shape)

fshift = util.firsttry(data)
fsrecon = util.shiftallto(data, fshift)

# for N=16 L=64 sigma=0.2 maxevd is deg=8
# for N=8 L=64 sigma=0.2 maxevd is deg=8

if False:
    deglist = range(5, 12)
    
    evdlist = []
    shiftlist = []
    coeflist = []
    
    start = datetime.now()
    for i, d in enumerate(deglist):
        print("deg = {}".format(d))
        reg = SuperRegistration(data, d, shifts=fshift) #shifts[1:])
        shiftlist.append(reg.fit(iprint=1, delta=1E-5, lamb=0.1))
        evdlist.append(reg.evidence(sigma=sigma))
        coeflist.append(reg.coef)
        print("evd = {}".format(evdlist[-1]))
    print("Calculation took {}".format(datetime.now()-start))
    
    today = format(datetime.now()).split()[0]
    reg = SuperRegistration(data, list(deglist)[np.argmax(evdlist)], 
                            shifts=shiftlist[np.argmax(evdlist)][0],
                            coef=coeflist[np.argmax(evdlist)])
else:
    reg = SuperRegistration(data, 8)
    reg.fit(iprint=1, delta=1e-5, lamb=0.1)

fsmog = makemog(fsrecon, pos, widths, sigma)
srmog = makemog(reg.model[0], pos, widths, sigma)
tmog = makemog(model[0], pos, widths, sigma)
pnames = list(fsmog.pmap.names)
pnames.remove('background')
pnames.remove('epsilon')
pnames.remove('theta')
p0 = fsmog.pmap.getvalues(pnames)
