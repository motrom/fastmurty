/**
Michael Motro github.com/motrom/fastmurty 4/2/19
*/
#include <string.h> // memcpy
#include <stddef.h> // size_t
#include "da.h" // many types and functions

/*
queue of problems
as well as storage for internal algorithms
the current structure of the data on heap is:
	current problem struct (pointers to problem data)
	current problem data
	queue (array of QueueEntries)
	array of [problem struct, problem data]
	--- separate ---
	workvarsforSSP
	--- separate ---
	workvarsforSplit

	Changing this internal structure shouldn't change the function of the code
	(changing the internal structure of Subproblem will as memcpy is used to copy parts)
*/
WorkvarsforDA allocateWorkvarsforDA(int m, int n, int nsols){
    WorkvarsforDA workvars;
	workvars.m = m;
	workvars.n = n;
	workvars.nsols = nsols;
	size_t solutionsize = sizeof(int)*(m + n) + sizeof(double)*n;
	size_t subproblemsize = solutionsize +
								sizeof(int)*(m + n) + sizeof(bool)*n;
	size_t fullsize = (subproblemsize + sizeof(Subproblem))*(nsols+1);
	fullsize += sizeof(QueueEntry)*nsols;
	char* buffer = (char*) malloc(fullsize);
	assert(buffer != NULL);
	workvars.buffer = buffer;
	workvars.currentproblem = (Subproblem*) buffer;
	buffer += sizeof(Subproblem);
	assert(buffer < workvars.buffer + fullsize);
	*workvars.currentproblem = allocateSubproblem(m, n, buffer);
	buffer += subproblemsize;
	workvars.Q = (QueueEntry *) buffer;
	// set up each subproblem's pointers
	QueueEntry* qele = workvars.Q;
	assert(buffer < workvars.buffer + fullsize);
	buffer += sizeof(QueueEntry)*nsols;
    int sol;
	Subproblem* prbele;
    for(sol=0; sol<nsols; sol++){
        assert(buffer < workvars.buffer + fullsize);
		prbele = (Subproblem*) buffer;
		qele->val = prbele;
		qele++;
		buffer += sizeof(Subproblem);
		assert(buffer < workvars.buffer + fullsize);
		*prbele = allocateSubproblem(m, n, buffer);
        buffer += subproblemsize;
    }
    workvars.sspvars = allocateWorkvarsforSSP(m, n);
    workvars.splitvars = allocateWorkvarsforSplit(m, n);
    return workvars;
};

void deallocateWorkvarsforDA(WorkvarsforDA workvars){
    deallocateWorkvarsforSSP(workvars.sspvars);
    deallocateWorkvarsforSplit(workvars.splitvars);
    free(workvars.buffer);
};



int da(inputmatrixtype c, int nrow_priors, bool* row_priors, double* row_prior_weights,
                    int ncol_priors, bool* col_priors, double* col_prior_weights,
                    int K, int* out_assocs, double* out_costs, WorkvarsforDA* workvars){
	const double inf = 1000000000;
	int m, n, m2, n2, m3, n3;
	int i, j, k, solidx, rowidx;
	WorkvarsforSSP* sspvars;
	WorkvarsforSplit* splitvars;
	QueueEntry* Q;
	int Qsize;
	QueueEntry bestsol, worstsol;
	int rowprior_idx, colprior_idx;
	double costbound;
	Subproblem* prb;
	Subproblem* bestprb;
	double C;
	size_t subproblemsize, solutionsize, eliminationsize;

    #ifndef NDEBUG
	    const double eps_debug = 0.00000001;
        bool passed = true;
	    int ri, ri2, cj, cj2;// id, jd, j2d, ;
	#endif


    m = workvars->m;
    n = workvars->n;
	if ((K == 0) || (nrow_priors == 0) || (ncol_priors == 0)) {
		for (k = 0; k < K; k++) {
			out_costs[k] = inf;
		}
		return 0;
	}
	for (j=0; j < K*(m+n)*2; j++){
	    out_assocs[j] = -2;
	}

	// prep variables
	subproblemsize = sizeofSubproblemData(m, n);
	solutionsize = sizeofSolutionData(m, n);
	eliminationsize = sizeof(bool)*n;

    // unpack workvars
    sspvars = &(workvars->sspvars);
    splitvars = &(workvars->splitvars);
    prb = workvars->currentproblem;
    for(j=0; j<n; j++){
        prb->eliminateels[j] = false;
    }
    prb->eliminatemiss = false;
    Q = workvars->Q;
    Qsize = K;
    // initial queue full of infinite costs
    for(solidx=0; solidx<Qsize; solidx++){
        Q[solidx].key = inf;
    }

    // find best solutions for each input hypothesis
    if (K==1) worstsol = Q[0]; else worstsol = Q[1];
    for(rowprior_idx=0; rowprior_idx < nrow_priors; rowprior_idx++){
        // partition so included rows are first
        m2 = 0;
        m3 = m;
        for(i=0; i<m; i++){
            if(row_priors[rowprior_idx*m + i]){
                prb->rows2use[m2] = i;
                m2++;
                prb->solution.x[i] = -1; //unnecessary?
            } else {
                m3--;
                prb->rows2use[m3] = i;
                prb->solution.x[i] = -2;
            }
        }
        prb->m = m2;
        for(colprior_idx=0; colprior_idx < ncol_priors; colprior_idx++){
            n2 = 0;
            n3 = n;
            for(j=0; j<n; j++){
                if(col_priors[colprior_idx*n + j]){
                    prb->cols2use[n2] = j;
                    n2++;
                    prb->solution.y[j] = -1;
                } else {
                    n3--;
                    prb->cols2use[n3] = j;
                    prb->solution.y[j] = -2;
                }
                prb->solution.v[j] = 0;
            }
            prb->n = n2;

            C = SSP(c, prb, sspvars);
            C += row_prior_weights[rowprior_idx];
            C += col_prior_weights[colprior_idx];
            if(C < worstsol.key){
                copySubproblem(worstsol.val, prb, subproblemsize);
                worstsol.key = C;
                worstsol = qReplaceMax(Q, worstsol, Qsize);
                
				#ifndef NDEBUG
				    for (ri = 0; ri < m; ri++) {
					    for (ri2 = 0; ri2 < ri; ri2++) {
					        assert(prb->rows2use[ri] != prb->rows2use[ri2]);
					    }
				    }
				    for (cj = 0; cj < n; cj++) {
					    for (cj2 = 0; cj2 < cj; cj2++) {
					        assert(prb->cols2use[cj] != prb->cols2use[cj2]);
					    }
				    }
                #endif
            }
        }
    }

    for(k=0; k<K; k++){
        Qsize = K-k-1;

        bestsol = qPopMin(Q, Qsize);
        if(bestsol.key >= inf){
            // less than K valid associations
            for(solidx=k; solidx<K; solidx++){
                out_costs[solidx] = inf;
            }
            return 1;
        }
        
        // copy solution to output
        out_costs[k] = bestsol.key;
        bestprb = bestsol.val;
        rowidx = k*(m+n)*2;
		for(j=0; j<n; j++){
		    i = bestprb->solution.y[j];
            if (i==-1){ // unmatched measurements first
                out_assocs[rowidx] = -1;
                rowidx++;
                out_assocs[rowidx] = j;
                rowidx++;
            }
        }
        for(i=0; i<m; i++){
            j = bestprb->solution.x[i];
            if (j!=-2){
                out_assocs[rowidx] = i;
                rowidx++;
                out_assocs[rowidx] = j;
                rowidx++;
            }
        }
        if(Qsize == 0) break;

        // prep for creating subproblems
        copySubproblem(prb, bestprb, subproblemsize);
        murtySplit(c, prb, splitvars);
        prb->n = splitvars->n_start;
        // set missing columns as uneliminated
        for(j = 0; j < n; j++){
            prb->eliminateels[j] = false;
        }
        j = 0;
        costbound = worstsol.key - bestsol.key;

        for(prb->m = splitvars->m_start+1; prb->m <= bestprb->m; prb->m++){
            copySolution(prb, bestprb, solutionsize);

            i = prb->rows2use[prb->m-1];
            prb->eliminateels[j] = false; // last subproblem's elimination
            prb->eliminatemiss = false;
            if(prb->m == bestprb->m){
                // this subproblem is originating problem w/ one more elimination
                // inherits originating problem's eliminations
                memcpy(prb->eliminateels, bestprb->eliminateels, eliminationsize);
                prb->eliminatemiss = bestprb->eliminatemiss;
            }
            j = prb->solution.x[i];
            if(j == -1){
                prb->eliminatemiss = true;
            } else {
				#ifndef NDEBUG
				    if (j != prb->cols2use[prb->n]) {
					    printf("ordering fail n=%d j=%d, cols[n]=%d\n",
					        prb->n, j, prb->cols2use[prb->n]);
					    return 1;
				    }
				#endif
                prb->eliminateels[j] = true;
                prb->n++;
            }

            // solve new subproblem
            C = spStep(c, prb, sspvars, costbound);
            if(C < costbound){
                copySubproblem(worstsol.val, prb, subproblemsize);
                worstsol.key = C + bestsol.key;
                worstsol = qReplaceMax(Q, worstsol, Qsize);
                costbound = worstsol.key - bestsol.key;
                
				#ifndef NDEBUG
				    assert(prb->m <= bestprb->m);
				    assert(prb->n <= bestprb->n);
				    assert(C > -eps_debug); // new solution should not be lower than previous best solution
				    for (ri = 0; ri < m; ri++) {
					    for (ri2 = 0; ri2 < ri; ri2++) {
					        assert(prb->rows2use[ri] != prb->rows2use[ri2]);
					    }
				    }
				    for (cj = 0; cj < n; cj++) {
					    for (cj2 = 0; cj2 < cj; cj2++) {
					        assert(prb->cols2use[cj] != prb->cols2use[cj2]);
					    }
				    }
			        if (C < inf){
				        // shouldn't get solution when an entire row has been eliminated
				        passed = false;
				        if (!prb->eliminatemiss){ passed = true; }
				        for(cj = 0; cj < prb->n; cj++){
				            if (!prb->eliminateels[prb->cols2use[cj]]){
				                passed = true;
				            }
				        }
				        assert(passed);
				    }
                 #endif
            }
        }
    }
    return 0;
};
