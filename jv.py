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
#import numba as nb

inf= 1e10 # inf is a suitably large number

#@nb.jit()
def LAPJVinit(c, x, y, v):
    """n: problem size, integer;
    c: costs matrix;
    x: columns assigned to rows;
    y: rows assigned to columns;"""
    #label augment;
    #var f,h,i,j,k, fO, il,jl,j2,ul,u2,min, last, low, up: integer;
    #col, d, free, pred: vec;
    """ col: array of columns, scanned (k = 1 ... low- 1),
                labeled and unscanned (k = low... up- 1),
                        unlabeled (k = up... n);
    d: shortest path lengths;
    free: unassigned rows (number f0, index f);
    pred: predecessor-array for shortest path tree;
    i, il : row indices; j, jl, j2: column indices;
    last: last column in col-array with d [j] <min.
    """
    n = c.shape[0]
    
    # COLUMN REDUCTION
    transfer_possible = np.zeros(n, np.bool8)

    for j in xrange(n-1,-1,-1):
        i = y[j]
        if x[i] < 0:
            x[i] = j
            transfer_possible[i] = True
        else:
            y[j] = -1
            
    # REDUCTION TRANSFER
    free = []
    for i,j1 in enumerate(x):
        if j1 < 0: # unassigned row in free-array
            free.append(i)
        elif transfer_possible[i]: # reduction transfer from assigned row
            original_val = v[j1]
            v[j1] = -inf
            v[j1] = original_val + np.max(v - c[i,:])
    f = len(free)
        
    # AUGMENTING ROW REDUCTION
    for cnt in xrange(2): # routine applied twice
        k = 0
        f0 = f
        f = 0
        while k < f0:
            i = free[k]
            k += 1
            u = c[i,:] - v
            j1 = np.argmin(u) # find to lowest indices and values
            u1 = u[j1]
            u[j1] = inf
            j2 = np.argmin(u)
            u2 = u[j2]

            i1 = y[j1]
            v[j1] += u1 - u2
            if u1 == u2 and i1 >= 0:
                # MM since first and second choice are equal
                # see if second column is unassigned
                j1 = j2
                i1 = y[j1]
            if i1 >= 0:
                # MM min column(s) already assigned
                if u1==u2:
                    free[f] = i1
                    f += 1
                else:
                    k -= 1
                    free[k] = i1
            x[i] = j1
            y[j1] = i
    return free[:f]
                   
        
# AUGMENTATION
#@nb.jit()
def LAPJVfinish(c, x, y, v, free):
    n = c.shape[0]
    col = np.arange(n)
    d = np.zeros((n,))
    pred = np.zeros(n, dtype=np.int64)
    
    for i1 in free:
        low = 0
        up = 0 # initialize d- and pred-array
        pred[:] = i1
        d[:] = c[i1,:] - v
        found_open_column = False
        while not found_open_column:
            if up == low: # find columns with new value for minimum d
                last = low# - 1
                minval = d[col[up]]
                up += 1
                for k in xrange(up,n):
                    j = col[k]
                    h = d[j]
                    if h <= minval + 1e-5:
                        if h < minval:
                            up = low
                            minval = h
                        col[k] = col[up]
                        col[up] = j
                        up += 1
                for j in col[low:up]:
                    if y[j]==-1:
                        found_open_column = True
                        break
            if found_open_column: break
                
            # scan a row
            j1 = col[low]
            low += 1
            i = y[j1]
            u1 = c[i,j1] - v[j1] - minval
            for k in xrange(up,n):
                j = col[k]
                h = c[i,j] - v[j] - u1
                if h < d[j]:
                    d[j] = h
                    pred[j] = i
                    ## MM self-implementation of isclose, bc numpy is slower
                    if abs(h-minval) < 1e-6*max(1, minval):
                        if y[j] == -1:
                            found_open_column = True
                            break
                        col[k] = col[up]
                        col[up] = j
                        up += 1
        
        # augment
        v[col[:last]] += d[col[:last]] - minval # updating of column prices
        # augmentation
        i = pred[j] # MM repeat instead of do-while
        y[j] = i
        k = j
        j = x[i]
        x[i] = k
        while i != i1:
            assert j >= 0
            assert i >= 0
            i = pred[j]
            y[j] = i
            k = j
            j = x[i]
            x[i] = k
            
#    # DETERMINE ROW PRICES AND OPTIMAL VALUE
#    u = c[range(n), x] - v
#    total_cost = sum(c[range(n), x])
    


if __name__ == '__main__':
    """
    create a random matrix
    try assignment, check for equality
    """
    from scipy.optimize import linear_sum_assignment
    
    m=7
    n=6
    P = np.random.exponential(size=(m,n))
    mX = np.random.exponential(size=(m,))
    mY = np.random.exponential(size=(n,))
#    P = np.random.rand(m,n)
#    mX = np.random.rand(m)
#    mY = np.random.rand(n)
    c = np.zeros((m+n,m+n))
    c[:m,:n] = P
    c[:m,n:] = 1e4
    c[range(m),range(n,m+n)] = mX
    c[m:,:n] = 1e4
    c[range(m,m+n), range(n)] = mY
    
    sol = linear_sum_assignment(c)
    x1 = list(sol[1][:m])
    y1 = range(n)
    for k in range(m+n):
        j = sol[1][k]
        if j < n:
            y1[j] = k
    cost = np.sum(c[sol[0],sol[1]])
    print x1
    print y1
    print cost
    
    y = np.argmin(c, axis=0)
    v = c[y, range(m+n)].copy()
    x = np.zeros((m+n,), dtype=int) - 1
    free = LAPJVinit(c, x, y, v)
    LAPJVfinish(c, x,y,v,free)
    cost = np.sum(c[range(m+n),x])
    print x[:m]
    print y[:n]
    print cost
    u = c[range(m+n),x] - v[x]
    slack = c - u[:,None] - v[None,:]
    assert np.min(slack) >= -1e-5 # might be slightly negative b.c. of float error