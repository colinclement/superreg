import sys
sys.path.append('../')
import numpy as np
from copy import deepcopy
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

import fourier_register as fr
from onedregister import Register


class BiasTest(object):
    def __init__(self, data_kwargs, **kwargs):
        self.data_kwargs = data_kwargs
        self.shifts = self.data_kwargs[1].get('shifts', np.random.randn(2))
        self.N = kwargs.pop('N', 1000)
        self.noises = kwargs.pop('noises', np.linspace(0.01, 0.15, 10))

        self._regkwargs = kwargs.copy()
        self.setdata(data_kwargs)
        self.reg = Register(self.data[1], self.data[0], **kwargs)

    def setdata(self, data_kwargs):
        args, kwargs = data_kwargs
        self.data, self.s, self.true, _ = fr.fakedata(*args, **kwargs)

    def getdata(self, noise):
        return self.true + noise * np.random.randn(*self.true.shape)

    def repeat(self, noise):
        p1s, p1_sigmas = [], []
        for i in range(self.N):
            img0, img1 = self.getdata(noise)
            p1, p1_sigma = self.reg.fit(imag1=img1, imag0=img0)
            p1s += [p1]
            p1_sigmas += [p1_sigma]
        return p1s, p1_sigmas

    def noiseloop(self):
        results = {'bias': [], 'biaserr': [], 'err': []}
        alldata = []
        for n in self.noises:
            p1s, p1_sigmas = self.repeat(n)
            alldata += [[p1s, p1_sigmas]]
            results['bias'] += [np.mean(p1s)-self.delta]
            results['biaserr'] += [np.std(p1s)/np.sqrt(len(p1s))]
            results['err'] += [np.mean(p1_sigmas)]
        return results, np.array(alldata)

    def deltaloop(self, deltas, noise=0.075):
        results = {'bias': [], 'biaserr': [], 'err': []}
        alldata = []
        dkwargs = deepcopy(self.data_kwargs)
        for d in deltas:
            dkwargs[1]['shifts'] = [d]
            self.setdata(dkwargs)
            p1s, p1_sigmas = self.repeat(noise)
            alldata += [[p1s, p1_sigmas]]
            results['bias'] += [np.mean(p1s)-d]
            results['biaserr'] += [np.std(p1s)/np.sqrt(len(p1s))]
            results['err'] += [np.mean(p1_sigmas)]
        return results, np.array(alldata)

    def plotbias(self, results, abscissa=None, xlabel=None, axis=None, title=None):
        biases = np.array(results['bias'])
        biases_std = np.array(results['biaserr'])
        err = np.array(results['err'])

        if axis is None:
            fig, axs = plt.subplots()
        else:
            axs = axis

        abscissa = abscissa if abscissa is not None else self.noises
        xlabel = xlabel if xlabel is not None else "Noise level $\sigma$"

        axs.errorbar(abscissa, biases,
                     yerr = biases_std, label=r"$\Delta$ bias",
                     linestyle="-", marker="o")
        axs.plot(abscissa, err, ':o', label=r"std$\Delta$")

        axs.set_xlabel(xlabel)
        axs.set_ylabel(r"$\langle\Delta\rangle - \Delta_\mathrm{true}$")
        if title: axs.set_title(title)
        axs.legend(loc='best')
        return plt.gcf(), axs
