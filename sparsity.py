#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import numba as nb

inf = 1e9

sparsedtype = np.dtype([('x', np.float64), ('idx', np.int64)])
nbsparsedtype = nb.from_dtype(sparsedtype)

def sparsify(c, s): # keep s lowest elements for each row
    c2 = np.zeros((c.shape[0],s), dtype=sparsedtype)
    for i, ci in enumerate(c):
        colsinorder = np.argsort(ci)
        c2[i]['idx'] = colsinorder[:s]
        c2[i]['x'] = ci[colsinorder[:s]]
    return c2
        
def unSparsify(c, n):
    m = c.shape[0]
    c2 = np.zeros((m,n)) + inf
    for i in xrange(m):
        c2[i, c[i]['idx']] = c[i]['x']
    return c2