# -*- coding: utf-8 -*-
"""
github.com/motrom/fastmurty last modified 5/17/19
a simple but inefficient implementation of HOMHT data association
used for testing the validity of the main code
very slow - don't use on anything bigger than 50x50!
Also, this code doesn't correctly handle the corner cases of empty input matrices
or all-miss associations.
"""
from scipy.optimize import linear_sum_assignment
from heapq import heappush, heappop
from itertools import chain

inf = 1e8

def da(c):
    miss = c>=0
    c = c.copy()
    c[miss] = 0
    solrow, solcol = linear_sum_assignment(c)
    matches = miss[solrow, solcol] == False
    solrow = solrow[matches]
    solcol = solcol[matches]
    cost = sum(c[solrow, solcol])
    assocs = chain(zip(solrow, solcol),
                   ((row,-1) for row in range(c.shape[0]) if row not in solrow),
                   ((-1,col) for col in range(c.shape[1]) if col not in solcol))
    return cost, assocs

def mhtda(c, row_priors, row_prior_weights, col_priors, col_prior_weights,
          out_assocs, out_costs, workvars=None):
        
    orig_c = c    
    Q = []
    out_assocs[:] = -2
    out_costs[:] = inf
    
    for row_set, row_set_weight in zip(row_priors, row_prior_weights):
        for col_set, col_set_weight in zip(col_priors, col_prior_weights):
            row_set = [row for row in range(orig_c.shape[0]) if row_set[row]]
            col_set = [col for col in range(orig_c.shape[1]) if col_set[col]]
            priorcost = row_set_weight + col_set_weight
            c = orig_c[row_set,:][:,col_set].copy()
            cost, assocs = da(c)
            assocs = tuple((row_set[row] if row>=0 else -1,
                       col_set[col] if col>=0 else -1) for row,col in assocs)
            cost += priorcost
            heappush(Q, (cost, priorcost, (), assocs, row_set, col_set, []))
    
    for solution in range(out_assocs.shape[0]):
        cost, priorcost, fixed_assocs, orig_assocs, row_set, col_set,\
                                        eliminate = heappop(Q)
        solution_assocs = sorted(fixed_assocs + orig_assocs)
        out_assocs[solution, :len(solution_assocs)] = solution_assocs
        out_costs[solution] = cost
        # murty's algorithm
        for thisrow, thiscol in orig_assocs:
            ###if thisrow == -1: continue
            # create altered version of the assignment problem
            c = orig_c.copy()
            thispriorcost = priorcost
            eliminate.append((thisrow,thiscol))
            for eliminaterow, eliminatecol in eliminate:
                if eliminaterow == -1:
                    c[:,eliminatecol] -= inf
                    thispriorcost += inf
                elif eliminatecol == -1:
                    c[eliminaterow,:] -= inf
                    thispriorcost += inf
                else:
                    c[eliminaterow,eliminatecol] += inf
            c = c[row_set,:][:,col_set]
            # solve altered version
            cost, assocs = da(c)
            assocs = tuple((row_set[row] if row>=0 else -1,
                       col_set[col] if col>=0 else -1) for row,col in assocs)
            cost += thispriorcost
            heappush(Q, (cost, thispriorcost, fixed_assocs, assocs,
                         row_set, col_set, eliminate))
            # fix this row and column for succeeding assignment problems
            col_set = list(col_set)
            row_set = list(row_set)
            fixed_assocs = fixed_assocs + ((thisrow, thiscol),)
            if thisrow == -1:
                col_set.remove(thiscol)
                eliminate = [(row,col) for row,col in eliminate if col!=thiscol]
            elif thiscol == -1:
                row_set.remove(thisrow)
                eliminate = [(row,col) for row,col in eliminate if row!=thisrow]
            else:
                priorcost += orig_c[thisrow, thiscol]
                row_set.remove(thisrow)
                col_set.remove(thiscol)
                eliminate = [(row,col) for row,col in eliminate if
                             row!=thisrow and col!=thiscol]
