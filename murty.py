#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Michael Motro, University of Texas at Austin
last modified 10/23/2018

the function murty() does all the work
the __main__ code at the bottom tests on random matrices

"""
import numpy as np
from jv import LAPJVinit, LAPJVfinish
from Queue import PriorityQueue

inf = 1e10 # sufficiently high float that will not cause NaNs in arithmetic

def murty(P, miss_X, miss_Y):
    m,n = P.shape
    if m == 0:
        if n > 0:
            yield (sum(miss_Y), tuple((-1,y) for y in range(n)))
        raise StopIteration
    elif n == 0:
        yield (sum(miss_X), tuple((x,-1) for x in range(m)))
        raise StopIteration

    # combining matching and missing scores into a single square matrix,
    # as suggested by MillerStoneCox
    # the bottom/right indices as auxiliary indices
    # P=[P00 P01 P02]    miss_X = [MX0 MX1]     miss_Y = [MY0 MY1 MY2]
    #   [P10 P11 P12]
    #         [P00 P01 P02 MX0 inf]
    #         [P10 P11 P12 inf MX1]
    #     c = [MY0 inf inf  0   0 ]
    #         [inf MY1 inf  0   0 ]
    #         [inf inf MY2  0   0 ]
    # More compact representations exist for the single best assignment,
    # and the k best assignments if miss costs are uniform.
    # I don't know of a better representation for varying miss costs, though.
    c = np.zeros((m+n,m+n))
    c[:m,:n] = P
    c[:m,n:] = inf
    c[range(m), range(n,m+n)] = miss_X
    c[m:,:n] = inf
    c[range(m,m+n), range(n)] = miss_Y
    
    # first solution
    # x,y,v,free = variables from Jonker-Volgenant algorithm
    # storing these post-initialization is optimization 1 from MillerStoneCox
    # x = matching column indices for each row, or -1 if there is no match atm
    # y = matching row indices for each column, or -1
    # v = dual variable (prices) for columns
    # free = list of unresolved columns
    y = np.argmin(c, axis=0)
    v = c[y, range(m+n)].copy()
    x = np.zeros((m+n,), dtype=int) - 1
    free = LAPJVinit(c, x, y, v)
    LAPJVfinish(c, x, y, v, free)

    # cost of this solution
    C = np.sum(c[range(x.shape[0]), x])
    # list of row+column pairs that are matched and removed from the current problem
    fixed = []
    # list of rows in the original problem that are still in the reduced problem
    reduced_x = tuple(range(m))
    # list of cols in the original problem that are still in the reduced problem
    reduced_y = tuple(range(n))
    # eliminated_i = -1 if the current problem has been fully solved
    # eliminated_i = i if the problem needs to be solved, prohibiting match [i,x[i]]
    eliminate_i = -1
    # prohibited matches - for rows/columns that are still in the reduced problem
    eliminated = []
    
    # sorted list insertion is slow, so priority queue is used
    # the numbers and tuples are stored first because they can be used to sort
    # sorting on lists or numpy arrays will raise an error
    Q = PriorityQueue()
    Q.put((C, eliminate_i, reduced_x, reduced_y, eliminated, x, v, fixed))
    
    
    while not Q.empty():
        C, eliminate_i, reduced_x, reduced_y, eliminated, x, v, fixed = Q.get()
        
        # convert reduced index tuples to lists, so they can be modified
        # and used to index numpy arrays
        reduced_x = list(reduced_x)
        reduced_y = list(reduced_y)
        # get reduced cost matrix
        m = len(reduced_x) # number of rows in reduced problem
        n = len(reduced_y) # number of columns in reduced problem
        c = np.zeros((m+n,m+n))
        c[:m,:n] = P[reduced_x,:][:,reduced_y]
        c[:m,n:] = inf
        c[range(m), range(n,m+n)] = miss_X[reduced_x]
        c[m:,:n] = inf
        c[range(m,m+n), range(n)] = miss_Y[reduced_y]
        for i,j in eliminated:
            c[i,j] = inf
        
        if eliminate_i >= 0:
            # unsolved problem
            # reconstruct y
            y = np.argsort(x)
            # eliminate the selected match
            j = x[eliminate_i]
            x[eliminate_i] = -1
            y[j] = -1
            c[eliminate_i,j] = inf
            eliminated.append((eliminate_i, j))
            # solve
            LAPJVfinish(c, x, y, v, [eliminate_i])
            Cnew = sum(c[range(m+n), x])
            Cnew += sum((P[i,j] if j>=0 else miss_X[i]) if i>=0 else miss_Y[j]
                        for i,j in fixed)
            # check that this is a valid solution - can be deleted to save time
            assert all(y[j]==i for i,j in enumerate(x))
            # check that the lower bound was in fact a lower bound - "  "
            assert Cnew - C > -1e-5
            if Cnew < inf:
                # put back in queue, will only send when it is the best solution
                Q.put((Cnew, -1, reduced_x, reduced_y, eliminated, x, v, fixed))
            
        else:
            # problem already solved, return solution
            # combine matches from reduced matrix with previously fixed matches
            output = [(reduced_x[i] if i<m else -1, reduced_y[j] if j<n else -1)
                        for i,j in enumerate(x) if i<m or j<n]
            output = tuple(sorted(fixed + output))
            yield (C, output)
        
            # create problems that don't include this solution
            u = c[range(m+n), x] - v[x]
            slack = c - u[:,None] - v[None,:]
            #assert np.all(slack >= -1e-5)
            slack[range(m+n),x] = inf
            while m > 0 and n > 0:
                # get the lower bound for each removed match
                # don't include the aux submatrix
                includex = [i for i,j in enumerate(x) if i<m or j<n]
                minslack_X = np.min(slack[includex,:], axis=1)
                minslack_Y = np.min(slack[:,x[includex]], axis=0)
                minslack_X = np.maximum(minslack_X, minslack_Y)
                # select the highest-cost subproblem next
                # optimization 3 from MillerStoneCox
                next_i = np.argmax(minslack_X)
                C_lowerbound = C + minslack_X[next_i]
                next_i = includex[next_i]
                next_j = x[next_i]
                if C_lowerbound < inf:
                    # add to queue, unsolved (optimization 2 from MillerStoneCox)
                    Q.put((C_lowerbound, next_i, tuple(reduced_x), tuple(reduced_y),
                           eliminated, x, v,
                           fixed))
                # copy arrays that will be altered
                eliminated = [ij for ij in eliminated]
                x = x.copy()
                v = v.copy()
                fixed = [ij for ij in fixed]
                
                # make reduced matrix for next successive problem
                # reducing the aux submatrix along with the main submatrix
                if next_i < m and next_j < n:
                    # reduce two rows and two columns
                    # one standard and one auxiliary for each
                    # there is a corresponding match in the aux submatrix
                    next_i_aux = next_j + m
                    next_j_aux = next_i + n
                    if x[next_i_aux] != next_j_aux:
                        # the aux submatrix actually doesn't match this row/col
                        # make an equivalent match, in order to remove them
                        current_aux_j = x[next_i_aux]
                        current_aux_i = next((i for i,j in enumerate(x)
                                                if j==next_j_aux), None)
                        assert current_aux_j >= n
                        assert current_aux_i >= m
                        x[current_aux_i] = current_aux_j
                        # maintain correct slack
                        slack[current_aux_i,current_aux_j] = inf
                    reduce_i = range(next_i) + range(next_i+1, next_i_aux) +\
                                range(next_i_aux+1, m+n)
                    reduce_j = range(next_j) + range(next_j+1, next_j_aux) +\
                                range(next_j_aux+1, m+n)
                    x = x[reduce_i]
                    x[x > next_j_aux] -= 1
                    x[x > next_j] -= 1
                    eliminated = [(i - (i>next_i_aux) - (i>next_i),
                                   j - (j>next_j_aux) - (j>next_j))
                                  for i,j in eliminated if
                                    i!=next_i and i!=next_i_aux and
                                    j!=next_j and j!=next_j_aux]
                    fixed_next_i = reduced_x.pop(next_i)
                    fixed_next_j = reduced_y.pop(next_j)
                    fixed.append((fixed_next_i, fixed_next_j))
                    m -= 1
                    n -= 1
                elif next_i < m:
                    # reduce a row and a corresponding auxiliary column
                    assert next_j == next_i + n
                    reduce_i = range(next_i) + range(next_i+1, m+n)
                    reduce_j = range(next_j) + range(next_j+1, m+n)
                    x = x[reduce_i]
                    x[x > next_j] -= 1
                    eliminated = [(i - (i>next_i), j - (j>next_j))
                                  for i,j in eliminated if
                                    i!=next_i and j!=next_j]
                    fixed_next_i = reduced_x.pop(next_i)
                    fixed.append((fixed_next_i, -1))
                    m -= 1
                elif next_j < n:
                    # reduce a column and corresponding auxiliary row
                    assert next_i == next_j + m
                    reduce_i = range(next_i) + range(next_i+1, m+n)
                    reduce_j = range(next_j) + range(next_j+1, m+n)
                    x = x[reduce_i]
                    x[x > next_j] -= 1
                    eliminated = [(i - (i>next_i), j - (j>next_j))
                                  for i,j in eliminated if
                                    i!=next_i and j!=next_j]
                    fixed_next_j = reduced_y.pop(next_j)
                    fixed.append((-1, fixed_next_j))
                    n -= 1
                slack = slack[reduce_i,:][:,reduce_j]
                v = v[reduce_j]

    raise StopIteration()
    
    
    
if __name__ == '__main__':
    from time import time
    
    M = 30
    N = 50
    n_solutions = 25
    n_repeats = 20
    
    totaltime = 0.
    allcosts = []
    for repeat in range(n_repeats):
        # uniform
        P = np.random.rand(M,N)
        # exponential
        P = -np.log(P)
        #P.tofile('test{:2d}.bin'.format(repeat))
        
        # consider missing rows and columns
        miss_X = P[1:,0]
        miss_Y = P[0,1:]
        P = P[1:,1:]
        # don't allow missing rows or columns, will only have solutions if M==N
        #miss_X = np.zeros((M,)) + inf
        #miss_Y = np.zeros((N,)) + inf

        costs = []  
        
        totaltime -= time()
        # actual operation here!
        assignment_iterator = murty(P, miss_X, miss_Y)
        for k in range(n_solutions):
            C,S = next(assignment_iterator)
            costs.append(C)
        totaltime += time()
        
        #np.save('msc{:2d}.npy'.format(repeat), costs)
    print(totaltime * (1000. / n_repeats)) # average runtime in milliseconds
