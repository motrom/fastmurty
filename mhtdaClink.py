# -*- coding: utf-8 -*-
"""
Michael Motro github.com/motrom/fastmurty 4/2/19
"""
import numpy as np
from ctypes import c_int, Structure, POINTER,\
                    RTLD_GLOBAL, CDLL, c_double, byref, c_char_p, c_bool

lib = CDLL("./mhtda.so", RTLD_GLOBAL)

sparse = True


""" c structures """
class Solution(Structure):
    _fields_ = [("x", POINTER(c_int)),
                ("y", POINTER(c_int)),
                ("v", POINTER(c_double))]
class Subproblem(Structure):
    _fields_ = [("buffer", c_char_p),
                ("m", c_int),
                ("n", c_int),
                ("rows2use", POINTER(c_int)),
                ("cols2use", POINTER(c_int)),
                ("eliminateels", POINTER(c_bool)),
                ("eliminatemiss", c_bool),
                ("solution", Solution)]
class QueueEntry(Structure):
    _fields_ = [("key", c_double), ("val", POINTER(Subproblem))]
class cs_di_sparse(Structure):
    _fields_ = [("nzmax", c_int),
                ("m", c_int),
                ("n", c_int),
                ("p", POINTER(c_int)),
                ("i", POINTER(c_int)),
                ("x", POINTER(c_double)),
                ("nz", c_int)]
if sparse:
    class PathTypessp(Structure):
        _fields_ = [("val", c_double),
                    ("i", c_int),
                    ("j", c_int)]
    class WVssp(Structure):
        _fields_ = [("Q", POINTER(PathTypessp)),
                    ("pathback", POINTER(c_int)),
                    ("m", c_int),
                    ("n", c_int)]
    class WVsplit(Structure):
        _fields_ = [("row_cost_estimates", POINTER(c_double)),
                    ("row_best_columns", POINTER(c_int)),
                    ("col_used", POINTER(c_bool)),
                    ("m", c_int),
                    ("n", c_int),
                    ("m_start", c_int),
                    ("n_start", c_int)]
    input_argtype = cs_di_sparse
else:
    class WVssp(Structure):
        _fields_ = [("distances", POINTER(c_double)),
                    ("pathback", POINTER(c_int)),
                    ("n", c_int)]
    class WVsplit(Structure):
        _fields_ = [("row_cost_estimates", POINTER(c_double)),
                    ("row_best_columns", POINTER(c_int)),
                    ("m", c_int),
                    ("n", c_int),
                    ("m_start", c_int),
                    ("n_start", c_int)]
    input_argtype = POINTER(c_double)
class WVda(Structure):
    _fields_ = [("buffer", c_char_p),
                ("m", c_int),
                ("n", c_int),
                ("nsols", c_int),
                ("solutionsize", c_int),
                ("subproblemsize", c_int),
                ("currentproblem", POINTER(Subproblem)),
                ("Q", POINTER(QueueEntry)),
                ("sspvars", WVssp),
                ("splitvars", WVsplit)]

""" c functions """
lib.da.argtypes = [input_argtype, c_int, POINTER(c_bool), POINTER(c_double),
                   c_int, POINTER(c_bool), POINTER(c_double),
                   c_int, POINTER(c_int), POINTER(c_double), POINTER(WVda)]
lib.da.restype = c_int

allocateWorkvarsforDA = lib.allocateWorkvarsforDA
allocateWorkvarsforDA.argtypes = [c_int, c_int, c_int]
allocateWorkvarsforDA.restype = WVda

deallocateWorkvarsforDA = lib.deallocateWorkvarsforDA
deallocateWorkvarsforDA.argtypes = [WVda]

lib.SSP.argtypes = [input_argtype, POINTER(Subproblem), POINTER(WVssp)]
lib.SSP.restype = c_double

allocateWorkvarsforSSP = lib.allocateWorkvarsforSSP
lib.allocateWorkvarsforSSP.argtypes = [c_int, c_int]
lib.allocateWorkvarsforSSP.restype = WVssp

lib.createSubproblem.argtypes = [c_int, c_int]
lib.createSubproblem.restype = Subproblem
lib.deallocateSubproblem.argtypes = [POINTER(Subproblem)]


""" handler functions """
def mhtda(c, row_sets, row_set_weights, col_sets, col_set_weights,
       out_assocs, out_costs, workvars):
    """
    feeds numpy array / sparse matrix input and output to mhtda C library
    """
    if sparse:
        c_c = c[0]
    else:
        c_c = c.ctypes.data_as(POINTER(c_double))
    row_sets_c = row_sets.ctypes.data_as(POINTER(c_bool))
    row_set_weights_c = row_set_weights.ctypes.data_as(POINTER(c_double))
    col_sets_c = col_sets.ctypes.data_as(POINTER(c_bool))
    col_set_weights_c = col_set_weights.ctypes.data_as(POINTER(c_double))
    out_assocs_c = out_assocs.ctypes.data_as(POINTER(c_int))
    out_costs_c = out_costs.ctypes.data_as(POINTER(c_double))
    nrowpriors = c_int(row_sets.shape[0])
    ncolpriors = c_int(col_sets.shape[0])
    nsols = c_int(out_assocs.shape[0])
    
    err = lib.da(c_c, nrowpriors, row_sets_c, row_set_weights_c,
               ncolpriors, col_sets_c, col_set_weights_c,
               nsols, out_assocs_c, out_costs_c, byref(workvars))
    assert err == 0

def SSP(c, workvars):
    """
    runs single best data association on numpy array or sparse matrix data
    """
    if sparse:
        c_c = c[0]
        m = c_c.m
        n = c_c.n
        assert m <= workvars.m
        assert n <= workvars.n
    else:
        m,n = c.shape
        assert n <= workvars.n
        c = np.pad(c, ((0,0),(0,workvars.n-n)), 'constant', constant_values = 0)
        c_c = c.ctypes.data_as(POINTER(c_double))
        
    x = np.zeros(m, dtype=np.int32) + 33
    y = np.zeros(n, dtype=np.int32)
    v = np.zeros(n)
    rows2use = np.arange(m, dtype=np.int32)
    cols2use = np.arange(n, dtype=np.int32)
    sol = Solution(x.ctypes.data_as(POINTER(c_int)),
                   y.ctypes.data_as(POINTER(c_int)),
                   v.ctypes.data_as(POINTER(c_double)))
    prb = Subproblem()
    prb.solution = sol
    prb.m = m
    prb.n = n
    prb.rows2use = rows2use.ctypes.data_as(POINTER(c_int))
    prb.cols2use = cols2use.ctypes.data_as(POINTER(c_int))
#    prb = lib.createSubproblem(m, n)
    lib.SSP(c_c, byref(prb), byref(workvars))
#    x = [prb.solution.x[i] for i in xrange(m)]
#    y = [prb.solution.y[j] for j in xrange(n)]
    return x, y
    
    
""" additional useful functions """

def sparsifyByRow(c, nvalsperrow):
    """
    creates a row-ordered sparse matrix with a fixed number of elements per row
    the lowest-valued elements are kept, still arranged in order of column value
    """
    m,n = c.shape
    nvalsperrow = min(n, nvalsperrow)
    nvals = m*nvalsperrow
    cp = np.arange(0, nvals+1, nvalsperrow, dtype=np.int32)
    ci = np.empty(nvals, dtype=np.int32)
    cx = np.empty(nvals, dtype=np.float64)
    for i, crow in enumerate(c):
        if nvalsperrow < n:
            colsbyvalue = np.argpartition(crow, nvalsperrow)
        else:
            colsbyvalue = np.arange(nvalsperrow)
        colsinorder = np.sort(colsbyvalue[:nvalsperrow])
        ci[i*nvalsperrow:(i+1)*nvalsperrow] = colsinorder
        cx[i*nvalsperrow:(i+1)*nvalsperrow] = crow[colsinorder]
    cstruct = cs_di_sparse(c_int(nvals), c_int(m), c_int(n),
                            cp.ctypes.data_as(POINTER(c_int)),
                            ci.ctypes.data_as(POINTER(c_int)),
                            cx.ctypes.data_as(POINTER(c_double)), c_int(nvals))  
    # have to return numpy arrays too, or they might get recycled
    return (cstruct, cp, ci, cx)
    
def sparsifyByElement(c, nvals):
    """
    creates a row-ordered sparse matrix with a fixed number of elements
    the lowest-valued elements are kept, in increasing order of value
    """
    m,n = c.shape
    nvals = min(m*n, nvals)
    c = c.flatten()
    elsbyvalue = np.argpartition(c, nvals)
    elsinorder = np.sort(elsbyvalue[:nvals])
    cp = np.searchsorted(elsinorder // n, np.arange(m+1)).astype(np.int32)
    ci = (elsinorder % n).astype(np.int32)
    cx = c[elsinorder].astype(np.float64)
    cstruct = cs_di_sparse(c_int(nvals), c_int(m), c_int(n), byref(cp),
                           byref(ci), byref(cx), c_int(nvals))
    # have to return numpy arrays too, or they might get recycled
    return (cstruct, cp, ci, cx)

import numba as nb
@nb.njit(nb.i8(nb.i8[:,:], nb.b1[:,:], nb.i4[:,:,:], nb.i8[:,:], nb.i8))
def processOutput(matches, hypotheses, out_assocs, backward_index, n_matches):
    """
    Transforms the pairs found by the data association algorithm to a more usable
    format for tracking: a vector of matches and a binary matrix of associations.
    Usually it is also necessary to only keep a fixed number of matches.
    This version removes matches that are found after the limit has been hit,
    without considering the relative probabilities of existence.
    A serious tracker will probably want a better one - i.e. summing hypothesis
    scores for each match to estimate total probabilities of existence.
    """
    nm = 0
    nsols = out_assocs.shape[0]
    matches[:] = -1
    backward_index[:] = -1
    hypotheses[:] = False
    for k in range(nsols):
        hypothesis = hypotheses[k]
        for rr in range(out_assocs.shape[1]):
            i,j = out_assocs[k,rr]
        #for i,j in out_assocs[k]:
            if i == -2: break
            backidx = backward_index[i,j]
            if backidx == -1:
                if n_matches == nm:
                    continue
                backward_index[i,j] = nm
                matches[nm] = (i,j)
                backidx = nm
                nm += 1
            hypothesis[backidx] = True
    return nm
