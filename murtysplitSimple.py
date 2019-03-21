#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 3/14/19
These functions reorder rows and columns before creating subproblems.
The goal is to set it up so the first subproblem fixes everything
but the first non-missing row.
One row and column is unfixed (w/ match or miss eliminated) every new problem.
"""

import numba as nb
from sparsity import nbsparsedtype

nbpairtype = nb.typeof((0,0))

# reorder rows so that misses are first
# last row should always remain last, so previous eliminations are kept
# reorder columns so that they are eliminated in order along with the rows
@nb.jit(nbpairtype(nbsparsedtype[:,:], nb.i8[:], nb.i8[:], nb.f8[:], nb.i8[:],
                nb.i8, nb.i8[:], nb.i8), nopython=True)
def murtySplit(c, x, y, v, rows2use, m2, cols2use, n2):
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