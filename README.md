# fastmurty
This code solves the k-best assignments problem. That is, given a matrix of costs it successively finds 1-to-1 assignments of rows to columns, in increasing order of the total cost of assigned elements. This problem occurs in, for instance, data association in multi-object tracking or sensor fusion.
Specifically, this is an optimized version of Murty's algorithm developed by Miller, Stone, and Cox [1]. Their version is significantly faster than the classic Murty's algorithm on average and in the worst case. Note that C code for [1] is publicly available from Cox's website [2]. This repo has two advantages:

1. The C code operates on square cost matrices - that is, it finds assignments between two sets of equal size. However, the paper [1] outlines a modification that can handle sets of varying size, with a 'miss cost' for elements from one set that are not matched to any in the other set. This code implements that modification.

2. The C code requests that  
"NEC Research Institute Inc. shall be given a copy of any such derivative work or modified version of the software and NEC Research Institute Inc. and its affiliated companies (collectively referred to as NECI) shall be granted permission to use, copy, modify and distribute the software for internal use and research."  
I'm not sure what that constitutes, but now I/you don't need to be!

## Dependencies
Python 2 with numpy. Use in Python 3 will some require modifications - the queue import, range/xrange, etc.

## Usage
The file has test code under \_\_main\_\_, which demonstrates its use. The function murty() is an iterator, meaning you get the next assignment by calling next(). Each returned assignment is a tuple (cost, solution). Solution is a tuple of matchings, and a matching is a tuple (i,j), where i and j are row and column indices. If a row is not matched to any columns, the matching will be (i,-1), and similarly for missing columns.

## Other sources
Other public implementations of the k-best assignments problem:

+ [3] uses the optimization for the worst-case runtime and is also not restricted to square matrices. C++ code and a Matlab interface is available at [4]. Even without the queueing optimizations, the difference in language makes [3]'s implementation much faster than mine, and most likely the best one for someone using those languages.
+ [5] and [6] provide binaries that implement k-best assignments, but don't have source code online and don't go into detail about the implementation.
+ [7-12] all implement the unoptimized version of Murty's algorithm.

## Timed test
Getting the 25 best assignments from 50x50 uniform-random cost matrices, with infinite miss cost (so that methods that don't consider misses can be compared).

| code | runtime |
|------|---------|
| [2]  |   3 ms  |
| [3]  |   7 ms  |
| this |  46 ms  |
| [8]  |  1.6 s  |

## Referenced
[1] Miller, M. L., et al. “Optimizing Murty’s Ranked Assignment Method.” IEEE Transactions on Aerospace and Electronic Systems, vol. 33, no. 3, July 1997, pp. 851–62.  
[2] https://ingemarcox.cs.ucl.ac.uk/?page_id=9  
[3] Crouse, David F. "On implementing 2D rectangular assignment algorithms." IEEE Transactions on Aerospace and Electronic Systems 52.4 (2016): 1679-1696.  
[4] https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary  
[5] www.multiplehypothesis.com  
[6] https://nmr.cit.nih.gov/xplor-nih/  
[7] https://code.google.com/archive/p/java-k-best/  
[8] https://github.com/sandipde/Hungarian-Murty  
[9] https://gist.github.com/ryanpeach/738b560fd903857c061063d25b3c8225  
[10] https://github.com/fbaeuerlein/MurtyAlgorithm  
[11] https://github.com/cosmo-epfl/glosim/blob/master/libmatch/lap/murty.py  
[12] http://ba-tuong.vo-au.com/codes.html