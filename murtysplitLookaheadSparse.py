#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 3/14/19
These functions reorder rows and columns before creating subproblems.
The goal is to set it up so the first subproblem fixes everything
but the first non-missing row.
One row and column is unfixed (w/ match or miss eliminated) every new problem.
"""

import numpy as np
import numba as nb
from sparsity import nbsparsedtype

nbpairtype = nb.typeof((0,0))

# reorder rows so that misses are first
# last row should always remain last, so previous eliminations are kept
# reorder columns so that they are eliminated in order along with the rows
@nb.jit(nbpairtype(nbsparsedtype[:,:], nb.i8[:], nb.i8[:], nb.f8[:], nb.i8[:],
                nb.i8, nb.i8[:], nb.i8), nopython=True)
def partitionDefault(c, x, y, v, rows2use, m2, cols2use, n2):
    m3 = 0 # number of missing rows
    for ri in xrange(m2-1):
        i = rows2use[ri]
        j = x[i]
        if j == -1: # missing row
            rows2use[ri] = rows2use[m3]
            rows2use[m3] = i
            m3 += 1
    if x[rows2use[m2-1]] == -1:
        m2 -= 1
    n3 = 0 # number of missing columns
    for cj in xrange(n2):
        j = cols2use[cj]
        if y[j] == -1:
            cols2use[cj] = cols2use[n3]
            cols2use[n3] = j
            n3 += 1
    assert n2-n3==m2-m3 # number of reported matches is the same
    cols2use[n3:n2] = x[rows2use[m3:m2]]
    # if there are missing columns, must eliminate on all rows
    # if no missing columns, can eliminate only matched rows
    return (0, n3) if n3 > 0 else (m3, 0)

            
@nb.jit(nbpairtype(nbsparsedtype[:,:], nb.i8[:], nb.i8[:], nb.f8[:], nb.i8[:],
                nb.i8, nb.i8[:], nb.i8, nb.f8[:], nb.i8[:], nb.i8[:]), nopython=True)
def murtySplit(c, x, y, v, rows2use, m2, cols2use, n2,
                       row_cost_estimates, row_best_columns, pred):
    if m2 <= 2 or n2 <= 1:
        return partitionDefault(c, x, y, v, rows2use, m2, cols2use, n2)
    
    pred[:] = 0
    pred[cols2use[:n2]] = 1
    
    # order missing columns at beginning, they will not be removed no matter
    # the partition order    
    n3 = 0 # number of missing columns
    for cj in xrange(n2):
        j = cols2use[cj]
        if y[j] == -1:
            cols2use[cj] = cols2use[n3]
            cols2use[n3] = j
            n3 += 1
    n_missing_cols = n3
    
    # set aside row m2-1 and its column
    last_column = x[rows2use[m2-1]]
    if last_column != -1:
        for cj in xrange(n2-1):
            j = cols2use[cj]
            if j == last_column:
                cols2use[cj] = cols2use[n2-1]
                cols2use[n2-1] = j
        n2 -= 1 # don't use this column in lookahead
        pred[last_column] = 0
    m2 -= 1
    
    # determine if all rows will be eliminated or not
    n_not_eliminated_rows = 0
    if n_missing_cols == 0:
        # in this case, you can keep missing rows at the beginning and not fix them
        m3 = 0 # number of missing rows
        for ri in xrange(m2):
            i = rows2use[ri]
            j = x[i]
            if j == -1: # missing row
                rows2use[ri] = rows2use[m3]
                rows2use[m3] = i
                m3 += 1
        assert m3 == m2 - n2
        n_not_eliminated_rows = m3
    
    # find estimated cost for row --- min(c'[i,j]) for j!=x[i]
    for ri in xrange(n_not_eliminated_rows, m2):
        i = rows2use[ri]
        j = x[i]
        ui = 0.
        minval = 1e3 if j==-1 else 0. # value of missing
        minj = -1
        for cij in c[i]:
            j2 = cij['idx']
            if pred[j2]:
                dj = cij['x'] - v[j2]
                if j2 == j:
                    ui = dj
                else:
                    if dj < minval:
                        minval = dj
                        minj = j2
        row_cost_estimates[ri] = minval - ui
        row_best_columns[ri] = minj
        
    n3 = n2
    for m3 in xrange(m2-1, n_not_eliminated_rows-1, -1):
        # choose the *worst* current row and partition on this *last*
        # meaning that partition has the fewest fixed rows & the most freedom
        worst_ri = np.argmax(row_cost_estimates[n_not_eliminated_rows:m3+1])
        worst_ri += n_not_eliminated_rows
        worst_i = rows2use[worst_ri]
        rows2use[worst_ri] = rows2use[m3]
        rows2use[m3] = worst_i
        # don't want to pick this row again, can just overwrite it
        row_cost_estimates[worst_ri] = row_cost_estimates[m3]
        row_best_columns[worst_ri] = row_best_columns[m3]
        
        deadj = x[worst_i]
        if deadj != -1:
            # swap columns so this particular column matches that row
            for cj in xrange(n3):
                j = cols2use[cj]
                if j == deadj:
                    cols2use[cj] = cols2use[n3-1]
                    cols2use[n3-1] = deadj
                    break
            pred[deadj] = 0
            n3 -= 1
            # update other cost estimates that had picked the same column
            for ri in xrange(n_not_eliminated_rows, m3):
                if row_best_columns[ri] == deadj:
                    # recalculate without deadj
                    i = rows2use[ri]
                    j = x[i]
                    ui = 0.
                    minval = 1e3 if j==-1 else 0. # value of missing
                    minj = -1
                    for cij in c[i]:
                        j2 = cij['idx']
                        if pred[j2]:
                            dj = cij['x'] - v[j2]
                            if j2 == j:
                                ui = dj
                            else:
                                if dj < minval:
                                    minval = dj
                                    minj = j2
                    row_cost_estimates[ri] = minval - ui
                    row_best_columns[ri] = minj
    return n_not_eliminated_rows, n_missing_cols