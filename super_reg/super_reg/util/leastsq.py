from __future__ import print_function, division
import numpy as np
from copy import deepcopy as copy
from scipy.linalg import solve, LinAlgError


class LM(object):
    def __init__(self, res, jac, args = ()):
        """
        Custom implementation of Levenburg-Marqardt.
    	input:
    	    res: function which takes parameter arrays of length M
    	        and optionally specified args
    	    jac: function which takes paramter arrays of length M
    	        and optionally specified args
    	    args: tuple of positional arguments to be passed into res
    	        and jac
    	"""
        self.res = res
        self.jac = jac
        self.args = args

        self.msgdict = {0: 'Failed number of steps reached maxtries',
                        1: 'Maximum number of iterations reached',
                        2: 'Convergence criterion satisfied'}

    def leastsq(self, p0, maxiter=40, delta=1E-6, accept=10., 
                reject=5., iprint=0, geo=True, **kwargs):
        """
        A custom implementation of the Levenburg Marquardt least-squares 
        optimization algorithm.
        input:
            p0: initial list of parameter values
            maxiter: maximum number of iterations
            delta: finishing condition, algorithm quits if logprobability
                    changes by less than delta
            accept: float, factor to increase lambda upon accepting a step
            reject: float, factor to decrease lambda upon rejecting a step
            geo: True/False, whether or not to use geodesic acceleration

            iprint: integer, 1 for few messages, 2 for more messages
            kwargs:
                maxtries: int, number of failed steps after which the
                algorithm gives up
                solver: str, either 'cg' for conjugate gradient
                or 'spsolver' for direct solver (faster but uses
                ENORMOUS amounts of memory for standard scipy spsolve)
        returns:
            opt_p (array of shape N_params), JTJ (Hessian), message (int)
            message = 0 if number of failed steps is maxtries
            message = 1 if the algorithm reach max iterations
            message = 2 is algorithm achieved convergence criterion
        """
        lamb = float(kwargs.get('lamb', 10.)) #start with small downward grad steps
        h = kwargs.get('h', 1E-5)
        maxtries = kwargs.get('maxtries', 200)
        truncerr = 0.

        p1 = copy(p0)
        for itn in range(maxiter):
            res0 = self.res(p1, *self.args)
            nlnprob0 = 0.5*res0.dot(res0)
            if iprint:
                print("Itn {}: nlnprob = {}".format(itn, nlnprob0))

            J = self.jac(p1, *self.args)
            JTJ = J.T.dot(J)
            JTr = J.T.dot(res0)
            
            success, tries, message = False, 0, 0
            while not success:
                if tries == maxtries:
                    self._report = [p1, trial, lamb]
                    return p1, JTJ.diagonal(), message
                try:
                    jtjdiag = JTJ.diagonal()
                    JTJ[np.diag_indices_from(JTJ)] += lamb
                    self.delta0 = -1 * solve(JTJ, JTr)

                    if geo:
                        self._rpp = 2./h*((self.res(p1+h*self.delta0) - res0)/h
                                          - J.dot(self.delta0))
                        self.delta1 = -0.5*solve(JTJ, J.T.dot(self._rpp))
                        truncerr = 2*(np.sqrt(self.delta1.dot(self.delta1))/
                                      np.sqrt(self.delta0.dot(self.delta0)))

                    JTJ[np.diag_indices_from(JTJ)] = jtjdiag

                except LinAlgError as er:
                    print("\tSingular matrix, lamb = {}".format(lamb))
                    JTJ[np.diag_indices_from(JTJ)] -= lamb
                    lamb *= reject
                    tries += 1
                    continue
                lamb = np.nan_to_num(lamb)

                trial = p1 + self.delta0

                res1 = self.res(trial, *self.args)
                nlnprob1 = 0.5*res1.dot(res1)
                 
                if geo and truncerr > 2:
                    success = False
                elif nlnprob1 < nlnprob0: #success
                    success = True
                else: #failed step
                    success = False 

                if success:
                    lamb /= accept
                    p1 = trial
                else:
                    lamb *= reject

                tries += 1
                if iprint > 1:
                    print("\tsuccess = {}, lnprob = {}, \
                          lamb = {}".format(success, nlnprob1, lamb))

            #Convergence condition    
            if (nlnprob0 - nlnprob1)/nlnprob0 < delta:
                if iprint:
                    print("log-prob relative changed by less than delta = {},\
                          optimum found".format(delta))
                message += 1
                break
        message += 1
        if iprint:
            print(self.msgdict[message])
        return p1, JTJ.diagonal(), message
