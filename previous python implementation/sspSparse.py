#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
The Jonker-Volgenant algorithm for finding the maximum assignment.

Michael Motro, University of Texas at Austin
last modified 3/5/2019

the __main__ code at the bottom tests this implementation, comparing it to
Scipy's linear_sum_assignment function. You'll need to have scipy in your
distribution to run this file on its own, but not to import it in other files.
"""
import numpy as np
import numba as nb
from sparsity import nbsparsedtype

inf= 1e9 # inf is a suitably large number


nbheapouttype = nb.typeof((0., 0, 0)) # float, int, int

heapdtype = np.dtype([('key', np.float64), ('i',np.int64), ('j',np.int64)])
nbheapdtype = nb.from_dtype(heapdtype)

@nb.jit(nb.void(nbheapdtype[:], nb.i8, nb.f8, nb.i8, nb.i8), nopython=True)
def heappush(heap, pos, newkey, newi, newj):
    while pos > 0:
        parentpos = (pos - 1) >> 1
        if newkey > heap[parentpos]['key']: break
        heap[pos] = heap[parentpos]
        pos = parentpos
    inputel = heap[pos]
    inputel['key'] = newkey
    inputel['i'] = newi
    inputel['j'] = newj
    
@nb.jit(nbheapouttype(nbheapdtype[:], nb.i8), nopython=True)
def heappop(heap, heapsize):
    minele = heap[0]
    minkey = minele['key']
    mini = minele['i']
    minj = minele['j']
    heapsize -= 1
    newele = heap[heapsize]
    newkey = newele['key']
    # Bubble up the smaller child until hitting a leaf.
    pos = 0
    childpos = 1    # leftmost child position
    while childpos < heapsize:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < heapsize:
            if heap[childpos]['key'] > heap[rightpos]['key']:
                childpos = rightpos
        if heap[childpos]['key'] > newkey:
            break
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = (pos<<1) + 1
    heap[pos] = newele
    return minkey, mini, minj


@nb.jit(nb.f8(nbsparsedtype[:,:], nb.i8[:], nb.i8[:], nb.f8[:],
                nb.i8[:], nb.i8, nb.i8[:], nb.i8, nbheapdtype[:], nb.i8[:]),
        nopython=True)
def SSP(c, x, y, v, rows2use, nrows2use, cols2use, ncols2use, d, pred):
    """ solves full 2D assignment problem
    c: matrix, sparse
    x: column indices that match to row, or -1 if row is missing
    y: match indices for column
    v: column reductions
    rows2use, nrows2use: rows in rows2use[:nrows2use] are considered part of the problem
    cols2use, ncols2use: " "
    d: workspace var, heap structure to store potential matches (pathcost, i, j)
    pred: workspace var, stores whether each column was used and path backward
    returns cost of assignment
    """
    n = y.shape[0]
    C = 0.
    y[:] = -1
    v[:] = 0.
    pred[:] = 0
    
    # initial step
    pred[cols2use[:ncols2use]] = -2
    freerows = 0
    for ri in xrange(nrows2use):
        i = rows2use[ri]
        minj = -1
        minval = 0.
        for cij in c[i]:
            j = cij['idx']
            if pred[j] == -2:
                val = cij['x']
                if val < minval:
                    minval = val
                    minj = j
        if minj == -1:
            x[i] = -1
            rows2use[ri] = rows2use[freerows]
            rows2use[freerows] = i
            freerows += 1
        elif y[minj] == -1:
            x[i] = minj
            y[minj] = i
            C += minval
            rows2use[ri] = rows2use[freerows]
            rows2use[freerows] = i
            freerows += 1
    
    for i1 in rows2use[freerows:nrows2use]:
        dsize = 0 # restart heap
        # pred doubles as a check for previously visited columns, and a path backwards
        pred[cols2use[:ncols2use]] = -2
        rowi = c[i1]
        for cij in rowi:
            j = cij['idx']
            dj = cij['x'] - v[j]
            if pred[j] == -2:
                heappush(d, dsize, dj, i1, j)
                dsize += 1
        heappush(d, dsize, 0., i1, -1)
        dsize += 1
        while True:
            predj = 0
            while predj != -2: # already found and matched this col
                assert dsize > 0
                minval, i, j = heappop(d, dsize)
                dsize -= 1
                if j == -1:
                    break # hit unmatched row
                predj = pred[j]
            if j == -1: break
            pred[j] = i
            v[j] += minval # first half of score augmentation
            i = y[j]
            if i == -1:
                break # hit unmatched column
            # update distances to other columns
            rowi = c[i]
            # find this row's reduction
            # have to look up the right column
            for cij in rowi:
                if cij['idx']==j:
                    u1 = cij['x'] - v[j]
            heappush(d, dsize, -u1, i, -1)
            dsize += 1
            for cij in rowi:
                j = cij['idx']
                if pred[j] == -2:
                    dj = cij['x'] - v[j] - u1
                    heappush(d, dsize, dj, i, j)
                    dsize += 1
        
        # augment
        # travel back through shortest path to find matches
        if j==-1:
            j = x[i]
            x[i] = -1
        while i != i1:
            i = pred[j]
            y[j] = i
            k = j
            j = x[i]
            x[i] = k
        # updating of column prices, part 2
        for j in xrange(n):
            if pred[j] != -2:
                v[j] -= minval
        # updating total cost
        C += minval
    return C
                
                
            
@nb.jit(nb.f8(nbsparsedtype[:,:], nb.i8[:], nb.i8[:], nb.f8[:], nb.i8[:],
              nb.i8, nb.i8[:], nb.i8, nbheapdtype[:], nb.i8[:], nb.i8, nb.i8,
              nb.b1[:], nb.b1, nb.f8), nopython=True)
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
    pred[:] = 0
    pred[cols2use[:ncols2use]] = -2 # set these as available
    dsize = 0
    rowi = c[i1]
    ui = 0.
    for cij in rowi:
        if cij['idx'] == j1:
            ui = cij['x'] - v[j1]
    for cij in rowi:
        j = cij['idx']
        if pred[j] == -2 and not eliminate_els[j]:
            dj = cij['x'] - v[j] - ui
            if dj <= cost_bound:
                heappush(d, dsize, dj, i1, j)
                dsize += 1
    if not eliminate_miss:
        dj = -ui
        if dj <= cost_bound:
            heappush(d, dsize, dj, i1, -1)
            dsize += 1
    minmissi = 0
    minmissj = 0
    miss_unused = True
    missing_from_row = False
    missing_cost = 0. # this is a dual cost on auxiliary columns
    while True:
        missing = False
        predj = 0
        while predj != -2: # already found and matched this col
            if dsize == 0:
                return inf
            minval, i, j = heappop(d, dsize)
            dsize -= 1
            if j == -1:
                if miss_unused:
                    minmissi = i
                    missing = True
                    missing_from_row = True
                    break # hit unmatched row
            else:
                predj = pred[j]
        if not missing:
            pred[j] = i
            v[j] += minval # first half of score augmentation
            if j==j1: break
            i = y[j]
            if i==-1:
                # entry to missing zone: col was missing but is now matched
                if miss_unused:
                    minmissj = j
                    missing = True
                    missing_from_row = False
                else:
                    # already covered the missing zone, this is a dead end
                    continue
        if missing:
            if j1 == -1:
                j=-1
                break
            miss_unused = False
            missing_cost = minval
            u1 = -minval
            # exit from missing zone: row that was missing is matched
            for i in rows2use[:nrows2use]:
                if x[i]==-1:
                    rowi = c[i]
                    for cij in rowi:
                        j = cij['idx']
                        if pred[j] == -2:
                            dj = cij['x']-v[j]-u1
                            if dj <= cost_bound:
                                heappush(d, dsize, dj, i, j)
                                dsize += 1
            # exit from missing zone: col that was matched is missing
            for j in cols2use[:ncols2use]:
                if y[j] >= 0 and pred[j] == -2:
                    dj = -v[j]-u1
                    if dj <= cost_bound:
                        heappush(d, dsize, dj, -1, j)
                        dsize += 1
        else:
            rowi = c[i]
            for cij in rowi:
                # first need to find this row's price
                # meaning we have to look up the right column
                if cij['idx']==j:
                    ui = cij['x'] - v[j]
            if miss_unused:
                dj = -ui
                if dj <= cost_bound:
                    heappush(d, dsize, dj, i, -1)
                    dsize += 1
            for cij in rowi:
                j = cij['idx']
                if pred[j] == -2:
                    dj = cij['x'] - v[j] - ui
                    if dj <= cost_bound:
                        heappush(d, dsize, dj, i, j)
                        dsize += 1
    
    # augment
    # travel back through shortest path to find matches
    i = i1+1 # any number that isn't i1
    while i != i1:
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
    # updating of column prices
    for j in cols2use[:ncols2use]:
        if pred[j]!=-2:
            v[j] -= minval
    if not miss_unused:
        v[cols2use[:ncols2use]] += minval - missing_cost
    v[y==-1] = 0.
    return minval




if __name__ == '__main__':
    """
    create a random matrix
    try assignment, check for equality
    """
    from scipy.optimize import linear_sum_assignment
    from sparsity import sparsedtype
    
    m = 10
    n = 20
    s = 10
#    P = np.random.exponential(size=(n,m))
#    mX = np.random.exponential(size=(n,))
#    mY = np.random.exponential(size=(m,))
    P = np.random.rand(m,n)*5
    mX = np.random.rand(m)
    mY = np.random.rand(n)
    
    # make sparse version
    # take s lowest columns from each row
    c2 = P - mX[:,None] - mY[None,:]
    csp = np.zeros((m,s), dtype=sparsedtype)
    for i in xrange(m):
        colsinorder = np.argsort(c2[i])
        csp[i]['idx'] = colsinorder[:s]
        csp[i]['x'] = c2[i, colsinorder[:s]]
        # make it sparse on the real matrices too
        P[i,colsinorder[s:]] = 100
        c2[i,colsinorder[s:]] = .01
    
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
    print(x1)
    print(y1)

    y = np.zeros(n, dtype=int) - 1
    x = np.zeros(m, dtype=int) - 1
    v = np.zeros(n)
    
    rows2use = np.arange(m)
    cols2use = np.arange(n)
    d = np.zeros(m*s+m, dtype=heapdtype)
    pred = np.zeros(n, dtype=int)
    C = SSP(csp, x, y, v, rows2use, cols2use, d, pred)
    print(x)
    print(y)
    
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
    C2 = sum(c2[i,j] for i,j in enumerate(x) if j>=0)
    assert np.isclose(C, C2)