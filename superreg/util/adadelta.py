"""
adadelta.py

author: Colin Clement
date: 2017-10-18

This is an implementation of the ADADELTA adaptic learning rate stochastic
gradient descent optimizer as described in https://arxiv.org/abs/1212.5701
"""

import numpy as np


class Adadelta(object):
    def __init__(self, obj_grad_obj, rho=0.9, eps=1e-6):
        """
        ADAM stochastic gradient descent optimizer.
        input:
            obj_grad_obj: Function which takes D (int) parameters
            and returns (objective, grad_objective), can take other args
            rho: exponential decay rate
            eps: regularization to prevent divide-by-zero
        """
        self.obj_grad_obj = obj_grad_obj
        self.rho = rho
        self.eps = eps
        
    def _reset(self, p0):
        self.t = 0
        self.obj_list = []
        self.m = np.zeros(len(p0))
        self.v = np.zeros(len(p0))
        
    def step(self, params, args = ()):
        """
        Take one step of ADAM SGD.
        input:
            params: array of D floats
            args: tuple of extra arguments to obj_grad_obj
        """
        obj, g = self.obj_grad_obj(params, *args)
        self.t += 1
        self.v[:] = self.rho * self.v + (1 - self.rho) * g * g
        delta = - np.sqrt((self.m + self.eps)/(self.v + self.eps)) * g
        self.m[:] = self.rho * self.m + (1 - self.rho) * delta * delta
        return delta, obj
    
    def optimize(self, p0, itn = 1000, tol = 1E-2,
                 iprint = 0, args = ()):
        """
        Run ADAM SGD.
        input:
            p0: array of D floats to start optimization
            itn : int number of iterations
            tol : change in objective below which algorithm terminates
            iprint : int for how often to print status of algorithm
            args : tuple of extra arguments to obj_grad_obj
        """
        self._reset(p0)
        obj0, _ = self.obj_grad_obj(p0, *args)
        for i in range(itn):
            delta, obj = self.step(p0, args)
            p0 += delta
            try:
                if i % iprint == 0:
                    print("Itn {:6d}: obj = {:8e}".format(i, obj))
            except ZeroDivisionError as perror:
                pass
            if np.abs((obj-obj0)/obj0) < tol:
                if iprint:
                    print("Relative change in objective less than tol")
                break
            self.obj_list += [obj]
            obj0 = obj
        return p0  
