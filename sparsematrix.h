/**
Michael Motro github.com/motrom/fastmurty 4/2/19
copied straight from cs.h in CXSparse package from SuiteSparse, http://www.suitesparse.com
*/
#ifndef _SPARSEMATRIX_H_
#define _SPARSEMATRIX_H_


typedef struct cs_di_sparse  /* matrix in compressed-column or triplet form */
{
	int nzmax;     /* maximum number of entries */
	int m;         /* number of rows */
	int n;         /* number of columns */
	int *p;        /* column pointers (size n+1) or col indices (size nzmax) */
	int *i;        /* row indices, size nzmax */
	double *x;     /* numerical values, size nzmax */
	int nz;        /* # of entries in triplet matrix, -1 for compressed-col */
} cs_di;

#endif
