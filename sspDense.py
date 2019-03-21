#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
The Jonker-Volgenant algorithm for finding the maximum assignment.

Michael Motro, University of Texas at Austin
last modified 10/23/2018

This is a direct adaptation of the Pascal code from
"A Shortest Augmenting Path Algorithm for Dense and Sparse Linear Assignment Problems"
R. Jonker and A. Volgenant, Computing 1987

the __main__ code at the bottom tests this implementation, comparing it to
Scipy's linear_sum_assignment function. You'll need to have scipy in your
distribution to run this file on its own, but not to import it in other files.
"""
import numpy as np
import numba as nb

inf= 1e9 # inf is a suitably large number

@nb.jit(nb.f8(nb.f8[:,:], nb.i8[:], nb.i8[:], nb.f8[:],
              nb.i8[:], nb.i8, nb.i8[:], nb.i8, nb.f8[:], nb.i8[:]), nopython=True)
def SSP(c, x, y, v, rows2use, nrows2use, cols2use, ncols2use, d, pred):
    """ solves full 2D assignment problem
    c: matrix
    x: column indices that match to row, or -1 if row is missing
    y: match indices for column
    v: column reductions
    rows2use, nrows2use: rows in rows2use[:nrows2use] are considered part of the problem
    cols2use, ncols2use: " "
    d, pred: workspace for SSP, remember costs and path backwards for each column
    returns cost of assignment
    """
    C = 0.
    
    # basic column reduction - basically running some rows in a convenient order
    nrows = nrows2use
    for ri in xrange(nrows2use-1,-1,-1):
        i = rows2use[ri]
        j = np.argmin(c[i,:])
        if c[i,j] < 0 and y[j] == -1:
            x[i] = j
            y[j] = i
            C += c[i,j]
            nrows -= 1
            rows2use[ri] = rows2use[nrows]
            rows2use[nrows] = i
    
    for i1 in rows2use[:nrows]:
        d[:] = c[i1,:] - v
        pred[:] = i1
        minmissi = i1
        minmissval = 0.
        ncolsunused = ncols2use
        emergcounter = 0
        while True:
            emergcounter += 1
            assert emergcounter < 2000
            minval = minmissval
            minj = -1
            mincolidx = 0
            for colidx, j in enumerate(cols2use[:ncolsunused]):
                dj = d[j]
                if dj < minval:
                    minj = j
                    minval = dj
                    mincolidx = colidx
            j = minj
            if j == -1:
                break # hit unmatched row
            i = y[j]
            if i == -1:
                break # hit unmatched column
            # this column should no longer be considered
            v[j] += minval
            ncolsunused -= 1
            cols2use[mincolidx] = cols2use[ncolsunused]
            cols2use[ncolsunused] = j
            # update distances to other columns
            u1 = c[i,j] - v[j]
            if -u1 < minmissval:
                # this row is the closest to missing
                minmissi = i
                minmissval = -u1
            for j in cols2use[:ncolsunused]:
                dj = c[i,j] - v[j] - u1
                if dj < d[j]:
                    d[j] = dj
                    pred[j] = i
        
        # augment
        # travel back through shortest path to find matches
        if j==-1:
            i = minmissi
            j = x[i]
            x[i] = -1
        emergcounter = 0
        while i != i1:
            emergcounter += 1
            assert emergcounter < 2000
            i = pred[j]
            y[j] = i
            k = j
            j = x[i]
            x[i] = k
        # updating of column prices
        for j in cols2use[ncolsunused:ncols2use]:
            v[j] -= minval
        C += minval
    return C
            


            
@nb.jit(nb.f8(nb.f8[:,:], nb.i8[:], nb.i8[:], nb.f8[:], nb.i8[:],
              nb.i8, nb.i8[:], nb.i8, nb.f8[:], nb.i8[:],
              nb.i8, nb.i8, nb.b1[:], nb.b1, nb.f8), nopython=True)
def spStep(c, x, y, v, rows2use, nrows2use, cols2use, ncols2use, d, pred,
           i1, j1, eliminate_els, eliminate_miss, cost_bound):
    """ solves Murty subproblem given solution to originating problem
    same inputs as SSP and also:
    i1, j1 = row and column that are now unassigned
    eliminate_els = boolean array, whether matching a column with i1 is prohibited
    eliminate_miss = whether i1 is prohibited to miss
    cost_bound = function will stop early and return inf if the solution is known
                    to be above this bound
    returns cost of shortest path, a.k.a. this solution's cost minus original solution's
    """
    
    if j1>=0:
        u0 = c[i1,j1]-v[j1] # not necessary to get solution, but gives accurate cost
    else:
        u0 = 0.
    pred[:] = i1
    ncols = ncols2use
    for j in cols2use[:ncols]:
        d[j] = inf if eliminate_els[j] else c[i1,j] - v[j] - u0
    minmissj = -1
    minmissi = i1
    minmissval = inf if eliminate_miss else -u0
    miss_unused = True
    missing_from_row = False
    missing_cost = 0. # this is a dual cost on auxiliary columns
    emergcounter = 0
    while True:
        emergcounter += 1
        assert emergcounter < 2000
        minval = minmissval
        minj = -2
        minjcol = -1
        for jcol, j in enumerate(cols2use[:ncols]):
            dj = d[j]
            if dj < minval:
                minj = j
                minval = dj
                minjcol = jcol
        if minval > cost_bound: return inf # that's all it takes for early stopping!
        j = minj
        if j==j1: break
        if j == -2:
            if not miss_unused: # if you got here again, costs must be really high
                return inf
            # entry to missing zone: row was matched but is now missing
            missing=True
            missing_from_row = True
        else:
            i = y[j]
            # this column should no lonber be considered
            ncols -= 1
            cols2use[minjcol] = cols2use[ncols]
            cols2use[ncols] = j
            if i==-1:
                # entry to missing zone: col was missing but is now matched
                if miss_unused:
                    minmissj = j
                    missing=True
                    missing_from_row = False
                else:
                    # already covered the missing zone, this is a dead end
                    continue
            else:
                missing=False
        if missing:
            if j1 == -1:
                j=-1
                break
            miss_unused = False
            missing_cost = minval
            minmissval = inf
            u1 = -minval
            # exit from missing zone: row that was missing is matched
            for i in rows2use[:nrows2use]:
                if x[i]==-1:
                    for j in cols2use[:ncols]:
                        dj = c[i,j]-v[j]-u1
                        if dj < d[j]:
                            d[j] = dj
                            pred[j] = i
            # exit from missing zone: col that was matched is missing
            for j in cols2use[:ncols]:
                if y[j] >= 0:
                    dj = -v[j]-u1
                    if dj < d[j]:
                        d[j] = dj
                        pred[j] = -1
        else:
            u1 = c[i,j]-v[j]-minval
            if miss_unused and -u1<minmissval:
                minmissi = i
                minmissval = -u1
            for j in cols2use[:ncols]:
                dj = c[i,j]-v[j]-u1
                if dj < d[j]:
                    d[j] = dj
                    pred[j] = i
    
    # augment
    # updating of column prices
    v[cols2use[ncols:ncols2use]] += d[cols2use[ncols:ncols2use]] - minval
    if not miss_unused:
        v[cols2use[:ncols2use]] += minval - missing_cost
    # travel back through shortest path to find matches
    i = i1+1 # any number that isn't i1
    emergcounter = 0
    while i != i1:
        emergcounter += 1
        assert emergcounter < 2000
        if j == -1:
            # exit from missing zone: row was missing but is now matched
            i = -1
        else:
            i = pred[j]
            y[j] = i
        if i == -1:
            # exit from missing zone: column j was matched but is now missing
            if missing_from_row:
                # entry to missing zone: row was matched but is now missing
                i = minmissi
                j = x[i]
                x[i] = -1
            else:
                # entry to missing zone: col was missing but is now matched
                j = minmissj
        else:
            k = j
            j = x[i]
            x[i] = k
    v[y==-1] = 0.
    return minval



if __name__ == '__main__':
    """
    create a random matrix
    try assignment, check for equality
    """
    from scipy.optimize import linear_sum_assignment
    
    m=10
    n=20
#    P = np.random.exponential(size=(n,m))
#    mX = np.random.exponential(size=(n,))
#    mY = np.random.exponential(size=(m,))
    P = np.random.rand(m,n)
    mX = np.random.rand(m)
    mY = np.random.rand(n)
    
    # make full square version, use standard code
    c1 = np.zeros((m+n,m+n))
    c1[:m,:n] = P
    c1[:m,n:] = 1e4
    c1[range(m),range(n,m+n)] = mX
    c1[m:,:n] = 1e4
    c1[range(m,m+n), range(n)] = mY
    sol = linear_sum_assignment(c1)
    x1 = np.array(sol[1][:m])
    x1[x1>=n] = -1
    y1 = np.arange(n)
    for k,j in enumerate(sol[1]):
        j = sol[1][k]
        if j < n:
            if k < m:
                y1[j] = k
            else:
                y1[j] = -1
    print x1
    print y1
    

    y = np.zeros(n, dtype=int) - 1
    x = np.zeros(m, dtype=int) - 1
    v = np.zeros(n)
    c2 = P - mX[:,None] - mY[None,:]
    rows2use = np.arange(m)
    cols2use = np.arange(n)
    d = np.zeros(n)
    pred = np.zeros(n, dtype=int)
    SSP(c2, x, y, v, rows2use, cols2use, d, pred)
    print x
    print y
    
    v += mY
    u = mX.copy()
    xmatch = x>=0
    xmis = xmatch==False
    ymis = y==-1
    u[xmatch] = P[xmatch,x[xmatch]] - v[x[xmatch]]
    u2 = np.append(u, np.zeros(n))
    v2 = np.append(v, np.zeros(m))
    x2 = np.append(x, y+n)
    x2[np.where(x==-1)[0]] = np.where(x==-1)[0]+n
    x2[np.where(y==-1)[0]+m] = np.where(y==-1)[0]
    slack = c1 - u2[:,None] - v2
    assert np.min(slack) > -1e-8
    assert all(slack[range(m+n), x2] < 1e-8)
    assert np.min(v[ymis]) >= -1e-8 if any(ymis) else True