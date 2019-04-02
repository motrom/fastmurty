/**
Michael Motro github.com/motrom/fastmurty 4/2/19
*/
#ifndef _DA_H_
#define _DA_H_

#include "queue.h"
// which modules are actually compiled is determined by the SPARSE preprocessor definition
#include "sspSparse.h"
#include "sspDense.h"
#include "murtysplitSparse.h"
#include "murtysplitDense.h"

/*
The necessary memory space is allocated separately from the function, in the expectation
that data association is usually performed repetitively, on problems of similar size.
*/
typedef struct WorkvarsforDAStruct {
    char* buffer;
	int m;
	int n;
	int nsols;
	Subproblem* currentproblem;
    QueueEntry* Q;
    WorkvarsforSSP sspvars;
    WorkvarsforSplit splitvars;
} WorkvarsforDA;

WorkvarsforDA allocateWorkvarsforDA(int m, int n, int nsols);
void deallocateWorkvarsforDA(WorkvarsforDA workvars);

/* Gets the K best associations for a problem.
c = either dense or sparse input matrix (only one is accepted, depends on definition SPARSE)
row_priors = [nrow_priors, nrows] matrix that specifies input hypotheses
    that is, subsets of rows to be considered
row_prior_weights = [nrow_priors] cost of each subset (added to cost of solution)
col_priors, col_prior_weights = same for columns
out_assocs = [K, nrows+ncols, 2], will be populated with output associations.
    out_assocs[k][z] stores the zth match of the kth best association - sorted by row.
    If the column value is -1 then the row was unmatched, and vice versa.
out_costs = [K], will be populated with the total cost of each association.
*/
int da(inputmatrixtype c, int nrow_priors, bool* row_priors, double* row_prior_weights,
                  int ncol_priors, bool* col_priors, double* col_prior_weights,
                  int K, int* out_assocs, double* out_costs, WorkvarsforDA* workvars);
                  
#endif
