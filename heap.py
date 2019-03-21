#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
last mod 3/2/19
inspired by 
"A Comparative Analysis of Three Different Priority Deques", Skov and Olsen
and the python 2.7 heapq
"""
import numpy as np
import numba as nb

heapdtype = np.dtype([('key', np.float64), ('val', np.int64)], align=False)
nbheapdtype = nb.from_dtype(heapdtype)
nbheapouttype = nb.typeof((0., 0)) # float, int

heapdtype_doubleidx = np.dtype([('key', np.float64), 
                                ('val', [('a',np.int64),('b',np.int64)])])

@nb.jit(nb.void(nbheapdtype[:], nb.i8, nb.f8, nb.i8), nopython=True)
def heappush(heap, pos, newkey, newval):
    while pos > 0:
        parentpos = (pos - 1) >> 1
        if newkey > heap[parentpos]['key']: break
        heap[pos] = heap[parentpos]
        pos = parentpos
    heap[pos]['key'] = newkey
    heap[pos]['val'] = newval
   
@nb.jit(nbheapouttype(nbheapdtype[:], nb.i8), nopython=True)
def heappop(heap, heapsize):
    minele = heap[0]
    minkey = minele['key']
    minval = minele['val']
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
    return minkey, minval

def heapIsValid(heap, heapsize):
    for pos in xrange(heapsize >> 1):
        leftpos = (pos<<1) + 1
        rightpos = leftpos + 1
        if heap[leftpos]['key'] < heap[pos]['key']:
            return False
        if rightpos < heapsize and heap[rightpos]['key'] < heap[pos]['key']:
            return False
    return True


@nb.jit(nb.void(nbheapdtype[:], nb.i8, nb.f8, nb.i8), nopython=True)
def iheappush(heap, pos, newkey, newval):
    if (pos & 1): # new count will be even, new element part of low-high pair
        lo = heap[pos-1]
        if newkey < lo['key']:
            # switch this pair
            heap[pos] = lo
            pos -= 1
            # move new pair up tree
            while pos > 1:
                parentpos = (pos - 1) >> 1 & -2 # lo index of parent of lo
                if newkey > heap[parentpos]['key']: break
                heap[pos] = heap[parentpos]
                pos = parentpos
        else:
            # new element is high part of pair
            # move new pair up tree
            while pos > 1:
                parentpos = (pos - 2) >> 1 | 1 # hi index of parent of hi
                if newkey < heap[parentpos]['key']: break
                heap[pos] = heap[parentpos]
                pos = parentpos
    else:
        # new count will be odd, this object is alone in new leaf
        parentlo = (pos - 1) >> 1 & -2
        if newkey < heap[parentlo]['key']:
            # new element belongs in the min heap
            while pos > 1:
                parentpos = (pos - 1) >> 1 & -2
                if newkey > heap[parentpos]['key']: break
                heap[pos] = heap[parentpos]
                pos = parentpos
        else:
            # might belong in max heap, or either
            while pos > 1:
                parentpos = (pos - 2) >> 1 | 1
                if newkey < heap[parentpos]['key']: break
                heap[pos] = heap[parentpos]
                pos = parentpos
    heap[pos]['key'] = newkey
    heap[pos]['val'] = newval
    
@nb.jit(nbheapouttype(nbheapdtype[:], nb.i8), nopython=True)
def iheappopmin(heap, heapsize):
    minele = heap[0]
    minkey = minele['key']
    minval = minele['val']
    heapsize -= 1
    newele = heap[heapsize]
    newkey = newele['key']
    newval = newele['val']
    pos = 0
    childpos = 2 # leftmost child position, lo index
    while childpos < heapsize:
        childkey = heap[childpos]['key']
        # Set childpos to index of smaller child.
        rightpos = childpos + 2
        if rightpos < heapsize:
            rightkey = heap[rightpos]['key']
            if rightkey < childkey:
                # branch to the right instead of the left
                childpos = rightpos
                childkey = rightkey
        if newkey < childkey:
            break # the new element is correctly positioned at pos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        # swap low and high if needed
        hichild = heap[childpos+1]
        if newkey > hichild['key']:
            newkey2 = hichild['key']
            newval2 = hichild['val']
            hichild['key'] = newkey
            hichild['val'] = newval
            # shifting the low index
            newkey = newkey2
            newval = newval2
        pos = childpos
        childpos = (pos<<1) + 2
    heap[pos]['key'] = newkey
    heap[pos]['val'] = newval
    return minkey, minval

@nb.jit(nb.void(nbheapdtype[:], nb.i8, nb.f8, nb.i8), nopython=True)
def iheapreplacemax(heap, heapsize, newkey, newval):
    pos = 1
    childpos = 3    # leftmost child position, lo index
    lochild = heap[0]
    if newkey < lochild['key']: # new element is smallest in heap
        newkey2 = lochild['key']
        newval2 = lochild['val']
        lochild['key'] = newkey
        lochild['val'] = newval
        # still shifting the high index
        newkey = newkey2
        newval = newval2
    size_limit = (heapsize-2)&-4
    while childpos < size_limit:
        childkey = heap[childpos]['key']
        # Set childpos to index of larger child.
        rightpos = childpos + 2
        rightkey = heap[rightpos]['key']
        if rightkey > childkey:
            # branch to the right instead of the left
            childpos = rightpos
            childkey = rightkey
        if newkey > childkey:
            break # the new element is correctly positioned at pos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        # swap low and high if needed
        lochild = heap[childpos-1]
        if newkey < lochild['key']:
            newkey2 = lochild['key']
            newval2 = lochild['val']
            lochild['key'] = newkey
            lochild['val'] = newval
            # still shifting the high index
            newkey = newkey2
            newval = newval2
        pos = childpos
        childpos = (pos<<1) + 1
    # address corner case at end of heap
    cornerstate = heapsize-childpos
    if cornerstate >= 0:
        # if cornerstate == 0, only lo index of left child is present
        leftpos = childpos-1 if cornerstate==0 else childpos
        if cornerstate == 2: # lo index of right child present
            rightpos = childpos+1
            if heap[rightpos]['key'] > heap[leftpos]['key']:
                childpos = rightpos
            else:
                childpos = leftpos
        else:
            childpos = leftpos
        if newkey < heap[childpos]['key']: # move down one more
            heap[pos] = heap[childpos]
            pos = childpos
            # check for swap
            if pos&1: # picked high index of left child
                lochild = heap[pos-1]
                if newkey < lochild['key']:
                    newkey2 = lochild['key']
                    newval2 = lochild['val']
                    lochild['key'] = newkey
                    lochild['val'] = newval
                    newkey = newkey2
                    newval = newval2
            
    heap[pos]['key'] = newkey
    heap[pos]['val'] = newval
    
@nb.jit(nbheapouttype(nbheapdtype[:], nb.i8), nopython=True)
def iheapgetmax(heap, heapsize):
    max_index = 0 if heapsize==1 else 1
    return heap[max_index]['key'], heap[max_index]['val']

def iheapIsValid(heap, heapsize):
    for pos in xrange(0, heapsize-1, 2):
        if heap[pos+1]['key'] < heap[pos]['key']:
            return False
    for pos in xrange(0, (heapsize-1) >> 1, 2):
        leftpos = (pos<<1) + 2
        rightpos = leftpos + 2
        if heap[leftpos]['key'] < heap[pos]['key']:
            return False
        if rightpos < heapsize and heap[rightpos]['key'] < heap[pos]['key']:
            return False
    for pos in xrange(1, heapsize>>1, 2):
        leftpos = (pos<<1) + 1
        rightpos = leftpos + 2
        if heap[leftpos]['key'] > heap[pos]['key']:
            return False
        if rightpos < heapsize and heap[rightpos]['key'] > heap[pos]['key']:
            return False
    if heapsize&1 and heapsize>1: # heap size is odd, one unpaired lo index
        if heap[(heapsize-2)>>1|1]['key'] < heap[heapsize-1]['key']:
            return False
    return True
    


    
if __name__ == '__main__':
    """
    Test using random pushes and pulls
    """
    import heapq
    np.random.seed(435)
    testsize = 40
    
    infloats = np.random.rand(testsize)
    inints = np.random.choice(range(testsize), size=testsize, replace=False)
    pushorpull = np.random.rand(testsize) > .25
    
    # test normal heap
    heap1 = []
    heap2 = np.zeros(testsize, dtype=heapdtype)
    heap2size = 0
    for test in range(testsize):
        if pushorpull[test] or len(heap1) == 0:
            heapq.heappush(heap1, (infloats[test], inints[test]))
            heappush(heap2, heap2size, infloats[test], inints[test])
            heap2size += 1
        else:
            out1 = heapq.heappop(heap1)
            out2 = heappop(heap2, heap2size)
            heap2size -= 1
            assert out1[0] == out2[0]
            assert out1[1] == out2[1]
        for j in xrange(len(heap1)):
            assert heap1[j][0] == heap2[j]['key']
            assert heap1[j][1] == heap2[j]['val']
        assert heapIsValid(heap2, heap2size)
            
#    # test interval heap
#    topk = 9
#    heap1 = []
#    heap2 = np.zeros(testsize, dtype=heapdtype)
#    heap2size = 0
#    for test in range(testsize):
#        if pushorpull[test]:
#            # push
#            if len(heap1) == topk:
#                topkval = max(heap1)
#                if topkval[0] > infloats[test]:
#                    h1 = heap1.index(topkval)
#                    heap1[h1] = (infloats[test], inints[test])
#                    heapq.heapify(heap1)
#            else:
#                heapq.heappush(heap1, (infloats[test], inints[test]))
#            if heap2size == topk:
#                topkval = heap2[1]['key']
#                if topkval > infloats[test]:
#                    iheapreplacemax(heap2, heap2size, infloats[test], inints[test])
#            else:
#                iheappush(heap2, heap2size, infloats[test], inints[test])
#                heap2size += 1
#        else:
#            if len(heap1) == 0:
#                continue
#            out1 = heapq.heappop(heap1)
#            out2 = iheappopmin(heap2, heap2size)
#            heap2size -= 1
#            assert out1[0] == out2[0]
#            assert out1[1] == out2[1]
#        assert iheapIsValid(heap2, heap2size)
#        assert len(heap1) == heap2size
#        if len(heap1) > 0:
#            match1 = np.array(sorted(heap1))
#            match2 = np.sort(heap2[:heap2size])
#            assert np.all(match1[:,0]==match2[:]['key'])
#            assert np.all(match1[:,1]==match2[:]['val'])
#            assert heap1[0][0] == heap2[0][0]
#            assert heap1[0][1] == heap2[0][1]
#        if len(heap1) > 1:
#            topkval = max(heap1)
#            assert topkval[0] == heap2[1][0]
#            assert topkval[1] == heap2[1][1]