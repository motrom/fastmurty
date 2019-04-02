/**
Michael Motro github.com/motrom/fastmurty 4/2/19
*/
#ifndef _SUBPROBLEM_H_
#define _SUBPROBLEM_H_

// libraries that most or all files use
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#ifndef NDEBUG
#include <stdio.h>
#endif

/* solution variable names follow Jonker-Volgenant's notation
 x[i] contains the matching column for row i, or -1 if there is no match
 y[j] contains the mathing row for column j, or -1 if there is no match
 u is the row reduction and v is the column reduction (only v is stored)
 u can be reconstructed given the input matrix -- u[i] = c[i,x[i]]-v[x[i]]
*/
typedef struct SolutionStruct {
	int *x;
	int *y;
	double *v;
} Solution;

typedef struct SubproblemStruct {
    // can be used to free data store later
	char *buffer;
	// number of rows and columns considered by problem
	int m;
	int n;
	// rows2use and cols2use are index vectors that reorder/partition the rows and
	// columns. This is useful as subsets of the rows or columns are taken several times
	// in the algorithm - when using multiple input hypotheses, when using Murty's
	//algorithm to split into subproblems, and during the shortest paths algorithm itself.
	int *rows2use;
	int *cols2use;
	// eliminated columns from a single row. As [] showed, if rows are processed in a
	// a smart order than eliminations will only appear in a single row, which is simpler
	// and cheaper than storing a list of eliminated elements across the matrix.
	bool *eliminateels;
	bool eliminatemiss;
	Solution solution;
} Subproblem;

Subproblem allocateSubproblem(int m, int n, char *buff);
Subproblem createSubproblem(int m, int n);
void deallocateSubproblem(Subproblem* prb);

size_t sizeofSubproblemData(int m, int n);
size_t sizeofSolutionData(int m, int n);
void copySubproblem(Subproblem* outprb, Subproblem* inprb, size_t size);
void copySolution(Subproblem* outprb, Subproblem* inprb, size_t size);

#endif
