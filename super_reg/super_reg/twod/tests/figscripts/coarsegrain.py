import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def crb(I, sigma):
    ly, lx = I.shape
    Ik = np.fft.fftn(I, norm='ortho')  # Unitary norm!
    ky = np.fft.fftfreq(ly, d=1./(2*np.pi))[:,None]
    kx = np.fft.fftfreq(lx, d=1./(2*np.pi))
    return sigma **2 / np.sum(ky**2 * Ik * Ik.conj()).real

def roughness(I):
    ly, lx = I.shape
    Ik = np.fft.fftn(I, norm='ortho')  # Unitary norm!
    ky = np.fft.fftfreq(ly, d=1./(2*np.pi))[:,None]
    return np.sum(ky**2 * Ik * Ik.conj()).real

def predictvar(I, sigma, N=None):
    s = crb(I, sigma)
    N = N or I.size
    return 2*s*(1+N*np.pi**2 * s/6)

def theory(D, sigma, N):
    s = sigma**2/D
    return 2*s*(1+N*np.pi**2 * s/6)

dat = np.load('results/coarsen-N_500-L_1024-2018-08-06.npz')
data = dat['data']
img = dat['datakwargs'][0]['img']
shift = dat['shift'].squeeze()
noises = dat['noises']
coarsenings = dat['coarsenings']

fig, axe = plt.subplots(1, 2, figsize=(12, 4.8))
for i, a in enumerate(coarsenings):
    axe[0].errorbar(noises, data.mean(1)[:,i,0,0]*a - shift[0], 
                    yerr=data.std(1)[:,i,0,0]*a/np.sqrt(500), 
                    label=str(a))
axe[0].set_title("Bias and coarsening")
axe[0].set_xlabel("Noise $\sigma$")

nn = np.linspace(noises.min(), noises.max(), 100)
D = roughness(img)
for i, (a, k) in enumerate(zip(coarsenings, mpl.rcParams['axes.prop_cycle'])):
    axe[1].plot(noises, data.std(1)[:,i,0,0]*a, 'o', label=str(a), **k)
    pvar = [np.sqrt(theory(D, n/a, (img.shape[0]/a)**2))*a for n in
            nn]
    axe[1].plot(nn, pvar, ':', label=str(a), **k)
axe[1].set_yscale('log')

plt.legend()
plt.show()
