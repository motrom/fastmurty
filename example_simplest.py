# -*- coding: utf-8 -*-
"""
Michael Motro github.com/motrom/fastmurty last modified 5/17/19

Runs single-input K-best associations algorithm on a simple matrix.
Intended to demonstrate usage.
"""

import numpy as np
from otherimplementations.slowmurty import mhtda as mhtdaSlow
from mhtdaClink import sparse, mhtda, allocateWorkvarsforDA
from mhtdaClink import sparsifyByRow as sparsify


cost_matrix = np.array([[-10, -9, -1],
                        [ -1, -6,  3],
                        [ -9, -5, -6]], dtype=np.float64)
nrows, ncolumns = cost_matrix.shape
nsolutions = 6 # find the 5 lowest-cost associations

# sparse cost matrices only include a certain number of elements
# the rest are implicitly infinity
# in this case, the sparse matrix includes all elements (3 columns per row)
cost_matrix_sparse = sparsify(cost_matrix, 3)
# The sparse and dense versions are compiled differently (see the Makefile).
# The variable "sparse" in mhtdaClink needs to match the version compiled
cost_matrix_to_use = cost_matrix_sparse if sparse else cost_matrix

# mhtda is set up to potentially take multiple input hypotheses for both rows and columns
# input hypotheses specify a subset of rows or columns.
# In this case, we just want to use the whole matrix.
row_priors = np.ones((1, nrows), dtype=np.bool8)
col_priors = np.ones((1, ncolumns), dtype=np.bool8)
# Each hypothesis has a relative weight too.
# These values don't matter if there is only one hypothesis...
row_prior_weights = np.zeros(1)
col_prior_weights = np.zeros(1)

# The mhtda function modifies preallocated outputs rather than
# allocating new ones. This is slightly more efficient for repeated use
# within a tracker.
# The cost of each returned association:
out_costs = np.zeros(nsolutions)
# The row-column pairs in each association:
# Generally there will be less than nrows+ncolumns pairs in an association.
# The unused pairs are currently set to (-2, -2)
out_associations = np.zeros((nsolutions, nrows+ncolumns, 2), dtype=np.int32)
# variables needed within the algorithm (a C function sets this up):
workvars = allocateWorkvarsforDA(nrows, ncolumns, nsolutions)

# run!
mhtda(cost_matrix_to_use,
      row_priors, row_prior_weights, col_priors, col_prior_weights,
      out_associations, out_costs, workvars)

# print each association
for solution in range(nsolutions):
    print("solution {:d}, cost {:.0f}".format(solution, out_costs[solution]))
    
    # display row-column matches, not row misses or column misses
    association = out_associations[solution]
    association_matches = association[(association[:,0]>=0) & (association[:,1]>=0)]
    mask = np.zeros((nrows, ncolumns), dtype=np.bool8)
    mask[association_matches[:,0], association_matches[:,1]] = True
    matrixstr = []
    for row in range(nrows):
        rowstr = []
        for column in range(ncolumns):
            elestr = ('{:3.0f}'.format(cost_matrix[row,column])
                        if mask[row,column] else ' * ')
            rowstr.append(elestr)
        matrixstr.append(' '.join(rowstr))
    print('\n'.join(matrixstr))
    
# compare to the simple(ish) Python implementation in slowmurty.py.
# There are a bunch of equivalent solutions, so the associations might not be
# in the same order.
out_costs2 = out_costs.copy()
mhtdaSlow(cost_matrix,
          row_priors, row_prior_weights, col_priors, col_prior_weights,
          out_associations, out_costs2, workvars)
assert np.allclose(out_costs, out_costs2)