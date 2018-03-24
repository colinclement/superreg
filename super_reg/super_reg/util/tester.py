import numpy as np
from copy import deepcopy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import super_reg.util.makedata as md


class BiasTest(object):
    def __init__(self, data_kwargs, registration, *args, **kwargs):
        """ args and kwargs are handed to registration """
        self.data_kwargs = data_kwargs
        self.N = kwargs.pop('N', 1000)
        self.noises = kwargs.pop('noises', np.linspace(0.01, 0.15, 10))

        self._regkwargs = kwargs.copy()
        self.setdata(data_kwargs)
        self.reg = registration(self.data, *args, **kwargs)

    def setdata(self, data_kwargs):
        self.data = md.fakedata(0., **data_kwargs)

    def getdata(self, noise):
        noisegen = self.data_kwargs.get('noisegen', np.random.randn)
        args0 = self.data.shape
        if 'noiseargs' in self.data_kwargs:
            args0 = args0 + self.data_kwargs['noiseargs']
        return self.data + noise * noisegen(*args0)

    def repeat(self, noise, **kwargs):
        p1s, p1_sigmas = [], []
        for i in range(self.N):
            p1, p1_sigma = self.reg.fit(self.getdata(noise), **kwargs)
            p1s += [p1]
            p1_sigmas += [p1_sigma]
        return p1s, p1_sigmas

    def noiseloop(self, **kwargs):
        alldata = []
        shifts = self.data_kwargs['shifts']
        for n in self.noises:
            p1s, p1_sigmas = self.repeat(n, **kwargs)
            alldata += [[p1s, p1_sigmas]]
        return np.array(alldata)

    def deltaloop(self, deltas, noise=0.075, **kwargs):
        alldata = []
        dkwargs = deepcopy(self.data_kwargs)
        for d in deltas:
            dkwargs['shifts'] = d
            self.setdata(dkwargs)
            p1s, p1_sigmas = self.repeat(noise, **kwargs)
            alldata += [[p1s, p1_sigmas]]
        return np.array(alldata)

    def plotbias(self, results, abscissa=None, xlabel=None, axis=None, title=None):
        biases = np.array(results['bias'])
        biases_std = np.array(results['biaserr'])
        err = np.array(results['err'])
        bias_std = np.array(results['bias_std'])

        if axis is None:
            fig, axs = plt.subplots()
        else:
            axs = axis

        abscissa = abscissa if abscissa is not None else self.noises
        xlabel = xlabel if xlabel is not None else "Noise level $\sigma$"

        axs.errorbar(abscissa, biases,
                     yerr = biases_std, label=r"$\Delta_y$ Bias",
                     linestyle="-", marker="o")
        axs.plot(abscissa, err, ':o', label=r"CRB of $\Delta_y$")
        axs.plot(abscissa, bias_std, ls='none', marker='D', label=r"Expected Error")

        axs.set_xlabel(xlabel)
        axs.set_ylabel(r"$\langle\Delta\rangle - \Delta_\mathrm{true}$")
        if title: axs.set_title(title)
        axs.legend(loc='best')
        return plt.gcf(), axs
