#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
github.com/motrom/fastmurty last mod 5/25/19
This is based on the pseudocode given in "Efficient Implementation..." by
B.N. Vo and B.T. Vo
This code is not necessarily a great implementation, it is primarily intended for
the comparison of Gibbs and deterministic algorithms in the paper.
Notably, the step of removing duplicate associations was not coded optimally.
That step is therefore not included in the paper's timing analysis.
That step could be performed in several ways... a hashtable by likelihood is
probably the fastest way but is difficult to create in Numba.
"""

import numpy as np
import numba as nb
from random import random

sparsedtype = np.dtype([('x', np.float64), ('idx', np.int64)])
nbsparsedtype = nb.from_dtype(sparsedtype)
def sparsify(c, s): # keep s lowest elements for each row
    c2 = np.zeros((c.shape[0],s), dtype=sparsedtype)
    for i, ci in enumerate(c):
        colsinorder = np.argsort(ci)
        c2[i]['idx'] = colsinorder[:s]
        c2[i]['x'] = ci[colsinorder[:s]]
    return c2

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
    
    for t in range(1, niter):
        for i in range(m):
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
    from scipy.optimize import linear_sum_assignment
    
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
    x = np.zeros(m, dtype=np.int64)
    out = np.zeros((100000, m), dtype=int)
    out_costs = np.zeros(100000)
    costs_struct = np.ones(m, dtype=np.float64)
    occupied_struct = np.zeros(n, dtype=np.bool8)
    for test in range(numtests):
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
            solrow, solcol = linear_sum_assignment(clog)
            x[:] = -1
            matches = clog[solrow,solcol] < 0
            x[solrow[matches]] = solcol[matches]
            timed_start = time()
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
    print(list(zip(niters, relcosts, runtimes)))