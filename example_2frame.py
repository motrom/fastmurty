# -*- coding: utf-8 -*-
"""
Michael Motro github.com/motrom/fastmurty last modified 4/2/19

Runs single-input K-best associations algorithm on square random matrices.
This test is meant to be directly comparable to the test code included with
Miller+Stone+Cox's implementation of data association.
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
    #max_val = -.1 # misses will occur (but are unlikely for large matrices)
    max_val = -float(size+1) # to ensure that misses are never picked
    timed_total = 0.
    relative_cost = 0.
    this_sparsity = min(30, size)
    workvars = allocateWorkvarsforDA(size, size, nsols)
    out_costs = np.zeros(nsols)
    out_associations = np.zeros((nsols, size*2, 2), dtype=np.int32)
    input_hypothesis = np.ones((1, size), dtype=np.bool8)
    input_score = np.zeros(1)
    
    for test in range(numtests):
        cd = np.random.rand(size, size) + max_val
        c = sparsify(cd, this_sparsity) if sparse else cd
        
        timed_start = time()
        mhtda(c, input_hypothesis, input_score, input_hypothesis, input_score,
                out_associations, out_costs, workvars)
        timed_end = time()
        timed_total += (timed_end-timed_start)
        relative_cost += sum(np.exp(-out_costs+out_costs[0]))
        
    my_results.append((timed_total*1000, relative_cost))
my_results = np.array(my_results) / numtests
print(my_results)
