# fastmurty
This code solves the data association problem for hypothesis-oriented multiple hypothesis tracking (HO-MHT). More generally this problem is the *k*-best 2D assignments or *k*-best bipartite matchings problem. That is, given a matrix of costs it successively finds 1-to-1 assignments of rows to columns, in increasing order of the total cost of assigned elements. The only alteration for the data association problem is that multiple subsets of rows and columns (representing prior hypotheses) may be considered.  
Murty's algorithm is a well-known solution to the *k*-best assignments problem, and implementations such as [1],[2] exist for data association. This implementation takes inspiration from those and adds some new optimizations. It is slightly faster than [1] and handles cases where an object doesn't have a matching measurement or vice versa (which is pretty important). For large and complicated problems, they are both considerably faster than [2] or any other implementations I'm aware of.  
A paper on the optimizations in question will be presented at the IEEE FUSION 2019 conference.

## Dependencies
The code was written in C with only standard library dependencies, for relatively easy porting to Python, MATLAB, C++, etc. A Python 2.7 port is included, and requires numpy. The Python-compiling package Numba is also used to reach high speeds in example_3frame.py, but can be removed without changing the functionality. Use in Python 3 may require some small modifications. The sparse input matrix is formatted with an equivalent structure to the column-ordered matrix of the CXSparse package, but CXSparse is not required.

## Usage
"make" will make a shared library file "mhtda.so" to be used in other programs. The expected use is:

    workvars = mhtda.allocateWorkvarsforDA(max_nrows, max_ncolumns, max_nassociations)
    err = mhtda.da(inputs..., outputs..., workvars)    /// err should be 0
    mhtda.deallocateWorkvarsforDA(workvars) /// optional

The exact form of all inputs and outputs is explained in da.h.  
There are two versions of the code, one which takes a normal cost matrix as input and one which tkaes a sparse matrix structure. Which version is compiled is determined by the "-D SPARSE" flag that can be turned on or off in the makefile. The Python link file has an example of creating a sparse matrix from a dense one.  
The other compilation option is "-D NDEBUG". If disabled, it will turn on several checks for solution accuracy. If you are suspicious of results on a certain problem, disable this option (and contact me if there is in fact a bug!).  
mhtdaClink.py uses Python's ctypes library to access these functions. The example files apply the algorithm to some simple problems. 2frame uses uniformly random matrices with a single input association, while 3frame sets up a 3D point-target sensor fusion problem, described in more detail in the file.  
There are also a few other codes in subfolders: a Python implementation using Numba that is similar to the main code in usage and speed, an MCMC data association (also Python with Numba), and slowmurty, a simple but not optimized or robust implementation built in Python with numpy/scipy.

## Other sources
Other public implementations of the k-best assignments problem:

+ C Code for [1] is available at [3].
+ C++ code and a Matlab interface for [2] are available at [4].
+ [5] and [6] provide binaries that implement k-best assignments, but don't have source code online and don't go into detail about the implementation.
+ [7-12] all implement the unoptimized version of Murty's algorithm, which has worst-case *O(kN^4)* runtime where *N* is the number of rows/columns. An optimized version is *O(kN^3)* and a highly sparse version is *O(kN^2)* (in typical cases runtime seems to be *O(kN)*).
+ [13] implements an unoptimized version but uses parallelization via OpenCL. There is a corresponding write-up that shows that their implementation is much faster with parallelization than without.
+ [14] is not free and doesn't describe its implementation (at least pre-purchase).
+ Stonesoup [15] is a new open-source tracking project. It doesn't have a MHT implementation yet. When it grows to incorporate C code I will ask if I can add this implementation in...

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
[13] https://github.com/nucleusbiao/Pedestrian-Tracking-using-SMC-LMB-with-OpenCL  
[14] https://www.mathworks.com/products/sensor-fusion-and-tracking.html  
[15] https://github.com/dstl/Stone-Soup