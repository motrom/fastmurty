#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 3/7/19
"""

import numba as nb

@nb.jit(nb.i8(nb.i8[:,:], nb.b1[:], nb.i8[:], nb.i8[:], nb.i8[:,:], nb.i8))
def processOutput(matches, hypothesis, x, y, backward_index, n_matches):
    """
    This one removes matches that are found after the limit has been hit,
    without considering the relative importance of each
    keeps all hypotheses
    """
    for i,j in enumerate(x):
        if j == -2: continue
        backidx = backward_index[i,j]
        if backidx == -1:
            if n_matches == matches.shape[0]:
                continue
            backward_index[i,j] = n_matches
            matches[n_matches] = (i,j)
            backidx = n_matches
            n_matches += 1
        hypothesis[backidx] = True
    for j,i in enumerate(y):
        if i==-1:
            backidx = backward_index[-1,j]
            if backidx == -1:
                if n_matches == matches.shape[0]:
                    continue
                backward_index[-1,j] = n_matches
                matches[n_matches] = (i,j)
                backidx = n_matches
                n_matches += 1
            hypothesis[backidx] = True
    return n_matches