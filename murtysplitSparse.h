/**
Michael Motro github.com/motrom/fastmurty 4/2/19
*/
#ifdef SPARSE

#include "subproblem.h"
#include "sparsematrix.h" // cs_di

typedef struct WorkvarsforSplitStruct {
	double* row_cost_estimates;
	int* row_best_columns;
	bool* col_used;
	int m;
	int n;
	int m_start;
	int n_start;
} WorkvarsforSplit;

WorkvarsforSplit allocateWorkvarsforSplit(int m, int n);
void deallocateWorkvarsforSplit(WorkvarsforSplit workvars);

/* reorders the rows and columns so that subproblem creation is simple.
The first subproblem fixes all matches except row 0 and column 0, the next unfixes
row 1 and column 1 (or just row 1 if it has no match), and so on.
This function reorders the rows so that the earlier, smaller subproblems are more likely,
using a lookahead estimate as described in the paper.
*/
void murtySplit(cs_di c, Subproblem* prb, WorkvarsforSplit* workvars);

#endif
