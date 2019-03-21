# fastmurty
This code solves the data association problem for hypothesis-oriented multiple hypothesis tracking (HO-MHT). More generally this problem is the *k*-best 2D assignments or *k*-best bipartite matchings problem. That is, given a matrix of costs it successively finds 1-to-1 assignments of rows to columns, in increasing order of the total cost of assigned elements. The only alteration for the data association problem is that multiple subsets of rows and columns (representing prior hypotheses) may be considered.  
Murty's algorithm is a well-known solution to the *k*-best assignments problem, and implementations such as [1],[2] exist for data association. This implementation takes inspiration from those and adds some new optimizations. It is slightly faster than [1] and handles cases where an object doesn't have a matching measurement or vice versa (which is pretty important). For large and complicated problems, they are both considerably faster than [2] or any other implementations I'm aware of.  
A paper on the optimizations in question has been submitted to the IEEE FUSION 2019 conference.

## Dependencies
Python 2 with numpy and numba. Numba can be removed without changing the functionality, but has a decimal-place impact on speed. Porting to Python 3 may require some modifications - range/xrange, etc.  
I plan to release a C version with a simple Python connector. Let's say within a month...

## Usage
daDense.py and daSparse.py each have a function da() that accept dense or sparse cost matrices as input. The sparsify() function from sparsity.py is an easy way to put a dense matrix into the right sparse format. da() also takes row and column subset arguments, pre-allocated output variables, and some workspace matrices. There are three output variables: out_matches is an array of (i,j) pairs representing matches between rows and columns, out_assocs is a binary matrix where each row is an association and each column corresponds to a row of out_matches, and out_costs is an array giving the total cost of each association in out_assocs. The workspace matrices can be made by the function allocateWorkVarsforDA(). They are allocated outside of the function because data association will be called repeatedly in a multi-object tracking algorithm.  
The example files show the functions used on some simple matrices. 2frame just uses uniformly random matrices with a single input association, while 3frame sets up Euclidean points in 3D space.

## Other sources
Other public implementations of the k-best assignments problem:

+ C Code for [1] is available at [3].
+ C++ code and a Matlab interface for [2] are available at [4].
+ [5] and [6] provide binaries that implement k-best assignments, but don't have source code online and don't go into detail about the implementation.
+ [7-12] all implement the unoptimized version of Murty's algorithm, which has worst-case *O(kN^4)* runtime where *N* is the number of rows/columns. An optimized version is *O(kN^3)* and a highly sparse version is *O(kN^2)* (and in typical cases *O(kN)*).

[1] Miller, M. L., et al. “Optimizing Murty’s Ranked Assignment Method.” IEEE Transactions on Aerospace and Electronic Systems, vol. 33, no. 3, July 1997, pp. 851–62.  
[2] Crouse, David F. "On implementing 2D rectangular assignment algorithms." IEEE Transactions on Aerospace and Electronic Systems 52.4 (2016): 1679-1696.  
[3] https://ingemarcox.cs.ucl.ac.uk/?page_id=9  
[4] https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary  
[5] www.multiplehypothesis.com  
[6] https://nmr.cit.nih.gov/xplor-nih/  
[7] https://code.google.com/archive/p/java-k-best/  
[8] https://github.com/sandipde/Hungarian-Murty  
[9] https://gist.github.com/ryanpeach/738b560fd903857c061063d25b3c8225  
[10] https://github.com/fbaeuerlein/MurtyAlgorithm  
[11] https://github.com/cosmo-epfl/glosim/blob/master/libmatch/lap/murty.py  
[12] http://ba-tuong.vo-au.com/codes.html