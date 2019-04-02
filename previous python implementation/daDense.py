#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import numba as nb
from sspDense import SSP, spStep
from heap import iheappopmin, iheapreplacemax, iheapgetmax, heapdtype
from murtysplitLookaheadDense import murtySplit
from outputSimple import processOutput

def allocateWorkVarsforDA(m, n, nsols):
    # following specify reduced problems, input to JV functions
    # keep a fixed bank of nsols problems, will replace unused problems as needed
    sols_rows2use = np.empty((nsols+1, m+1), dtype=int)
    sols_rows2use[:] = np.arange(m+1)
    sols_cols2use = np.empty((nsols+1, n+1), dtype=int)
    sols_cols2use[:] = np.arange(n+1)
    sols_elim = np.zeros((nsols+1, n+1), dtype=bool)
    sols_x = np.zeros((nsols+1,m), dtype=int)
    sols_v = np.zeros((nsols+1,n))
    backward_index = np.full((m+1,n+1), -1, dtype=int)
    return sols_rows2use, sols_cols2use, sols_elim, sols_x, sols_v, backward_index
            


@nb.jit(nb.void(nb.f8[:,:], nb.b1[:,:], nb.f8[:], nb.b1[:,:], nb.f8[:],
                nb.i8[:,:], nb.b1[:,:], nb.f8[:],
                nb.i8[:,:], nb.i8[:,:], nb.b1[:,:], nb.i8[:,:], nb.f8[:,:], nb.i8[:,:]),
        nopython=True)
def da(c, row_sets, row_set_weights, col_sets, col_set_weights,
       out_matches, out_assocs, out_costs,
       sols_rows2use, sols_cols2use, sols_elim, sols_x, sols_v, backward_index):
    """
    c: input matrix
    row_sets: nR*M binary matrix, specifies multiple input cases (for instance hypotheses)
    row_set_weights: nR float array, the cost of each row set (hypothesis weight)
    col_sets, col_set_weights: same for columns
        solving data association across different measurement subsets is uncommon
        but possible, see "Handling of Multiple Measurement Hypotheses..."
        by Kellner & Aeberhard, 2018
    out_matches = Nout*2 array of row-column matches that were used in associations
                    (i,-1) is a "missing" row, (-1,j) is a missing column
    out_assocs = K*Nout binary matrix, each row is an association and includes
                certain row-column matches
    out_costs = K float array, the cost of each output association
    sols_ = workspace variables
    """
    inf = 1e9 # sufficiently high float that will not cause NaNs in arithmetic
    m,n = c.shape
    nsols = out_assocs.shape[0]
    if m == 0 or nsols == 0 or len(row_sets)==0:
        out_costs[:] = inf
        return
    if n == 0 or len(col_sets)==0:
        out_assocs[0,:] = -1
        out_costs[0] = 1
        out_costs[1:] = inf
        return
    
    # reset output
    backward_index[:] = -1
    out_matches[:] = -1
    out_assocs[:] = False
    out_matches_n = 0
    # create some smaller temporary variables, for SSP and partitioning
    y = np.full(n, -1, dtype=np.int64)
    orig_y = y.copy()
    d = np.zeros(n)
    pred = np.zeros(n, dtype=np.int64)
    x = np.full(m, -1, dtype=np.int64)
    v = np.zeros(n)
    rows2use = np.arange(m)
    cols2use = np.arange(n)
    eliminate_els = np.zeros(n, dtype=np.bool8)
    partition_row_cost = np.zeros(m)
    partition_row_col = np.zeros(m, dtype=np.int64)
    # priority queue
    Q = np.zeros(nsols, dtype=heapdtype)
    for jj in range(nsols):
        Q[jj]['key'] = inf
        Q[jj]['val'] = jj
    Qsize = nsols
    
    # find best solutions for each input hypothesis
    cost_bound, solidx = iheapgetmax(Q, Qsize)
    for row_set_idx in xrange(len(row_sets)):
        row_set = row_sets[row_set_idx]
        m2 = 0 # partition so included set is first
        for i, in_set in enumerate(row_set):
            if in_set:
                rows2use[m2] = i
                m2 += 1
        m3=m2
        for i, in_set in enumerate(row_set):
            if not in_set:
                rows2use[m3] = i
                m3 += 1
        for col_set_idx in xrange(len(col_sets)):
            cols2use[:] = np.arange(n)
            col_set = col_sets[col_set_idx]
            n2 = 0 # partition so included set is first
            for i, in_set in enumerate(col_set):
                if in_set:
                    cols2use[n2] = i
                    n2 += 1
            n3 = n2
            for i, in_set in enumerate(col_set):
                if not in_set:
                    cols2use[n3] = i
                    n3 += 1
            x[rows2use[:m2]] = -1
            x[rows2use[m2:]] = -2
            y[cols2use[:n2]] = -1
            y[cols2use[n2:]] = -2
            v[:] = 0.
            C = SSP(c, x, y, v, rows2use, m2, cols2use, n2, d, pred)
            C += row_set_weights[row_set_idx]
            C += col_set_weights[col_set_idx]
            if C < cost_bound:
                sols_rows2use[solidx,:m2] = rows2use[:m2]
                sols_rows2use[solidx,m] = m2
                sols_cols2use[solidx,:n2] = cols2use[:n2]
                sols_cols2use[solidx,n] = n2
                sols_elim[solidx,:] = False
                sols_x[solidx] = x
                sols_v[solidx] = v
                iheapreplacemax(Q, Qsize, C, solidx)
                cost_bound, solidx = iheapgetmax(Q, Qsize)

    for k in xrange(nsols):
        Qsize = nsols-k-1 # current length of queue
        
        # get best solution from queue
        C, solidx = iheappopmin(Q, Qsize+1)
        if C >= inf: break
        orig_x = sols_x[solidx]
        n2 = sols_cols2use[solidx,n]
        cols2use[:n2] = sols_cols2use[solidx,:n2]
        # reconstruct y from x
        orig_y[:] = -2
        orig_y[cols2use[:n2]] = -1
        for i,j in enumerate(orig_x):
            if j >= 0:
                orig_y[j] = i
        # add to output
        out_costs[k] = C
        out_matches_n = processOutput(out_matches, out_assocs[k], orig_x, orig_y,
                                      backward_index, out_matches_n)
        if k == nsols-1:
            break
        
        # prep for creating subproblems
        m2 = sols_rows2use[solidx,m]
        rows2use[:m2] = sols_rows2use[solidx,:m2]
        orig_eliminate_els = sols_elim[solidx,:n]
        orig_eliminate_miss = sols_elim[solidx,n]
        orig_v = sols_v[solidx]
        # reorder rows2use and cols2use so that subproblem creation is simple
        m3_start, n3 = murtySplit(c, orig_x, orig_y, orig_v,
                                 rows2use, m2, cols2use, n2,
                                 partition_row_cost, partition_row_col)
        
        cost_bound, solidx = iheapgetmax(Q, Qsize)
        cost_bound = cost_bound - C
        
        for m3 in xrange(m3_start+1, m2+1):
            # subproblem
            # reset previous solution
            x[:] = orig_x
            v[:] = orig_v
            y[:] = orig_y
            
            # eliminate the selected match
            eliminate_i = rows2use[m3-1]
            eliminate_j = x[eliminate_i]
            if m3 == m2:
                # last elimination, mind eliminated cols from original
                eliminate_els[:] = orig_eliminate_els
                eliminate_miss = orig_eliminate_miss
            else:
                # undo elimination from previous problem
                eliminate_els[:] = False # should just be able to reset cols2use[n3]
                eliminate_miss = False
            if eliminate_j >= 0:
                n3 += 1 # column n3 is no longer fixed
                eliminate_els[eliminate_j] = True
            else:
                eliminate_miss = True
            
            # solve new problem
            # to remove early stopping, replace cost_bound with inf
            Cnew = spStep(c, x, y, v, rows2use, m3, cols2use, n3, d, pred,
                         eliminate_i, eliminate_j, eliminate_els, eliminate_miss,
                         cost_bound)
                         #inf)
            if Cnew < cost_bound:
                # add solution
                sols_rows2use[solidx,:m3] = rows2use[:m3]
                sols_rows2use[solidx,m] = m3
                sols_cols2use[solidx,:n3] = cols2use[:n3]
                sols_cols2use[solidx,n] = n3
                sols_elim[solidx,:n] = eliminate_els
                sols_elim[solidx,n] = eliminate_miss
                sols_x[solidx] = x
                sols_v[solidx] = v
                iheapreplacemax(Q, Qsize, C+Cnew, solidx)
                cost_bound, solidx = iheapgetmax(Q, Qsize)
                cost_bound = cost_bound - C
    

    
if __name__ == '__main__':
    from time import time
    np.random.seed(42)
    
    m = 20
    n = 20
    nout = 100
    nsols = 100
    n_repeats = 30
    
    totaltime = 0.
    
    row_sets = np.ones((2,m), dtype=bool)
    row_sets[1,0] = False
    row_set_weights = np.array([2.,0.])
    col_sets = np.ones((1,n), dtype=bool)
    col_set_weights = np.array([0.])
    
    workvars = allocateWorkVarsforDA(m, n, nsols)
    sols_rows2use, sols_cols2use, sols_elim, sols_x, sols_v, backward_index = workvars
    out_matches = np.zeros((nout, 2), dtype=int)
    out_assocs = np.zeros((nsols, nout), dtype=bool)
    out_costs = np.zeros(nsols)
    
    for repeat in range(n_repeats):
        # uniform
        c = np.random.rand(m,n)*5 - 3
        
        # consider missing rows and columns
        costs = []
        if repeat == 0:
            # run once to compile numba
            da(c, row_sets, row_set_weights, col_sets, col_set_weights,
                  out_matches, out_assocs, out_costs,
                  sols_rows2use, sols_cols2use, sols_elim, sols_x, sols_v, backward_index)
        
        totaltime -= time()
        # actual operation here!
        da(c, row_sets, row_set_weights, col_sets, col_set_weights,
              out_matches, out_assocs, out_costs,
              sols_rows2use, sols_cols2use, sols_elim, sols_x, sols_v, backward_index)
        totaltime += time()
        
    print(totaltime * (1000. / n_repeats)) # average runtime in milliseconds