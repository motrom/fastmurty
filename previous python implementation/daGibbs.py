#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This is based on the pseudocode given in "Efficient Implementation..." by
B.N. Vo and B.T. Vo
The step of removing duplicate associations is not included in the timing analysis.
This step could be performed in several ways... a hashtable by likelihood is
probably the fastest way but is difficult to create in Numba.
"""

import numpy as np
import numba as nb
from random import random
from sspSparse import nbsparsedtype


@nb.njit(nb.void(nbsparsedtype[:,:], nb.i8[:], nb.i8[:,:], nb.f8[:], nb.i8,
              nb.f8[:], nb.b1[:]))
def gibbs(c, x, out, out_costs, niter, costs, occupied):
    """ x is the best solution, obtained by sspSparse or some other method """
    m = c.shape[0]

    costs[:] = 1
    occupied[:] = False
    for i,j in enumerate(x):
        for cij in c[i]:
            if cij['idx'] == j:
                costs[i] = cij['x']
                occupied[j] = True
    out[0] = x
    out_costs[0] = np.prod(costs)
    missprob = 1.
    
    for t in xrange(1, niter):
        for i in xrange(m):
            j1 = x[i]
            if j1 != -1:
                occupied[j1] = False
                costs[i] = 1
            rowi = c[i]
            probsum = missprob
            for cij in rowi:
                if not occupied[cij['idx']]:
                    probsum += cij['x']
            choice = random() * probsum
            if choice < missprob:
                x[i] = -1
            else:
                probsum = missprob
                for cij in rowi:
                    j = cij['idx']
                    if not occupied[j]:
                        probsum += cij['x']
                        if choice < probsum:
                            x[i] = j
                            occupied[j] = True
                            costs[i] = cij['x']
        out[t] = x
        out_costs[t] = np.prod(costs)


@nb.njit(nb.f8(nb.i8[:,:], nb.f8[:], nb.i8[:]))
def getTotalCost(out, out_costs, order):
    totalcost = out_costs[order[-1]]
    lastx = out[order[-1]]
    for idx in order[-2::-1]:
        x = out[idx]
        if not np.all(x == lastx): # check for duplicate associations
            totalcost += out_costs[idx]
            lastx = x
    return totalcost




if __name__ == '__main__':
    """
    Determines the runtime and quality of this data association method.
    Note that this method really can't handle cases with very unlikely misses.
    The input matrices are set to be mostly positive for this reason.
    """
    from time import time
    from sspSparse import LAPJV, sparsify, heapdtype
    
    np.random.seed(23)
    numtests = 100
    m = 100
    n = 100
    sparsity = 10
    niters = np.logspace(1, 4, 4, base=10, dtype=int)
    
    
    my_results = []
    runtimes = np.zeros(len(niters))
    costs = np.zeros(len(niters))
    relcosts = np.zeros(len(niters))
    x = np.zeros(m, dtype=int)
    y = np.zeros(n, dtype=int)
    pred = np.zeros(n, dtype=int)
    v = np.zeros(n)
    d = np.zeros(m*sparsity+m, dtype=heapdtype)
    rows2use = np.arange(m)
    cols2use = np.arange(n)
    out = np.zeros((100000, m), dtype=int)
    out_costs = np.zeros(100000)
    costs_struct = np.ones(m, dtype=np.float64)
    occupied_struct = np.zeros(n, dtype=np.bool8)
    for test in xrange(numtests):
        # random matrix
        clog = np.random.rand(m,n) - .05
#        # 'geometric' matrix - detecting 1d points with white noise
#        pts = np.sort(np.random.rand(m))
#        ptnoise = np.random.normal(size=m)*.001
#        clog = np.square(pts[:,None] - pts[None,:] - ptnoise[None,:]) - .02
        clogs = sparsify(clog, sparsity)
        c = clogs.copy()
        c[:]['x'] = np.exp(-clogs[:]['x'])
        
        for k, niter in enumerate(niters):
            out[:] = 0
            timed_start = time()
            LAPJV(clogs, x, y, v, rows2use, m, cols2use, n, d, pred)
            gibbs(c, x, out, out_costs, niter, costs_struct, occupied_struct)
            timed_end = time()
            runtimes[k] += (timed_end-timed_start)
            outorder = np.argsort(out_costs[:niter])
            cost = getTotalCost(out, out_costs, outorder)
            costs[k] += np.log(cost)
            relcosts[k] += cost / out_costs[0]
            assert outorder[-1] == 0
            
    runtimes *= 1000 / numtests
    costs /= numtests
    relcosts /= numtests
    print zip(niters, relcosts, runtimes)