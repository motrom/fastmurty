# -*- coding: utf-8 -*-
"""
Michael Motro github.com/motrom/fastmurty last modified 5/16/19

Runs single-input K-best associations algorithm on a square random matrix,
then treats the result as an input hypothesis for a multiple-input K-best
associations algorithm on a rectangular random matrix.
"""

import numpy as np
from time import time
from mhtdaClink import sparse, mhtda, allocateWorkvarsforDA
from mhtdaClink import sparsifyByRow as sparsify

np.random.seed(0)
numtests = 100
nsols = 200
sizes = np.arange(10, 301, 10)
sparsity = 20 # elements per row

my_results = []
for size in sizes:
    print("running size {:d}".format(size))
    #max_val = 0. # misses will occur (but are unlikely for large matrices)
    max_val = -float(size+1) # to ensure that misses are never picked
    timed_total = 0.
    relative_cost = 0.
    this_sparsity = min(sparsity, size)
    size2 = size*(this_sparsity+2)
    workvars = allocateWorkvarsforDA(size, size, nsols)
    workvars2 = allocateWorkvarsforDA(size2, size, nsols)
    out_costs = np.zeros(nsols)
    out_costs2 = np.zeros(nsols)
    out_associations = np.zeros((nsols, size*2, 2), dtype=np.int32)
    out_associations2 = np.zeros((nsols, size2+size, 2), dtype=np.int32)
    input_hypothesis = np.ones((1, size), dtype=np.bool8)
    input_score = np.zeros(1)
    backidx = np.zeros((size+1,size+1), dtype=int)
    second_hypotheses = np.zeros((nsols, size2), dtype=np.bool8)
    
    for test in range(numtests):
        cd = np.random.rand(size, size) + max_val
        c1 = sparsify(cd, this_sparsity) if sparse else cd
        
        cd = np.random.rand(size2, size) + max_val
        c2 = sparsify(cd, this_sparsity) if sparse else cd    
            
        out_associations[:] = -2
        mhtda(c1, input_hypothesis, input_score, input_hypothesis, input_score,
                out_associations, out_costs, workvars)
        backidx[:] = -1
        matches = np.unique(out_associations.reshape((nsols*size*2, 2)), axis=0)
        matches = matches[matches[:,0] > -2]
        backidx[matches[:,0],matches[:,1]] = np.arange(matches.shape[0])
        second_hypotheses[:] = False
        for solution in range(nsols):
            association = out_associations[solution]
            association = association[association[:,0] > -2]
            association = backidx[association[:,0], association[:,1]]
            assert np.all(association >= 0) and np.all(association < size2) and association.shape[0]==size
            second_hypotheses[solution, association] = True
            
        timed_start = time()
        mhtda(c2, second_hypotheses, out_costs, input_hypothesis, input_score,
              out_associations2, out_costs2, workvars2)
        timed_end = time()
        timed_total += (timed_end-timed_start)
        
    my_results.append(timed_total*1000)
my_results = np.array(my_results) / numtests
print(my_results)
