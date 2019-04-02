# -*- coding: utf-8 -*-
"""
Runs single-input K-best associations algorithm on square random matrices.
This test is meant to be directly comparable to the test code included with
Miller+Stone+Cox's implementation of data association.
"""

import numpy as np
from time import time
from daSparse import da, allocateWorkVarsforDA
from sparsity import sparsify

np.random.seed(23)
numtests = 100
nsols = 200
sizes = np.arange(10, 301, 10)
sparsity = 30

my_results = []
for size in sizes:
#    max_val = -.1 # misses will occur (but are unlikely for large matrices)
    max_val = -float(size+1) # to ensure that misses are never picked
    noutsamples = size*5
    timed_total = 0.
    relative_cost = 0.
    this_sparsity = min(30, size)
    workvars = allocateWorkVarsforDA(size, size, nsols)
    out_matches = np.zeros((noutsamples, 2), dtype=int)
    out_associations = np.zeros((nsols, noutsamples), dtype=bool)
    out_costs = np.zeros(nsols)
    input_hypothesis = np.ones((1, size), dtype=bool)
    input_score = np.zeros(1)
    
    for test in xrange(numtests):
        cd = np.random.rand(size, size) + max_val
        c = sparsify(cd, this_sparsity)
        timed_start = time()
        da(c, input_hypothesis, input_score, input_hypothesis, input_score,
           out_matches, out_associations, out_costs, *workvars)
        timed_end = time()
        timed_total += (timed_end-timed_start)
        relative_cost += sum(np.exp(-out_costs+out_costs[0]))
    my_results.append((timed_total*1000, relative_cost))
my_results = np.array(my_results) / numtests
print(my_results)