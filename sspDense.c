/**
Michael Motro github.com/motrom/fastmurty 4/2/19
*/
#ifndef SPARSE

#include "sspDense.h"

WorkvarsforSSP allocateWorkvarsforSSP(int m, int n) {
	WorkvarsforSSP workvars;
    int totalsize = (sizeof(double) + sizeof(int)) * n;
	workvars.distances = (double *) malloc(totalsize);
	workvars.pathback = (int *)(workvars.distances + n);
	workvars.n = n;
	return workvars;
};

void deallocateWorkvarsforSSP(WorkvarsforSSP workvars) {
	free(workvars.distances);
};

double SSP(double* c, Subproblem* prb, WorkvarsforSSP* workvars){
    //const double inf = 1000000000;
	double C = 0;
	int cj, ri, i, i1, j, j1, minj, minmissi, mincol, rowidx, m, n;
    double minval, val, minmissval, ui;
	Solution sol = prb->solution;
	#ifndef NDEBUG
	const int loopescape = 10000;
	int loopcounter = 0;
	#endif

    // reset y, v
    for(cj=0; cj<prb->n; cj++){
        j = prb->cols2use[cj];
        sol.y[j] = -1;
        sol.v[j] = 0;
    }

    // basic column reduction - basically running some rows in a convenient order
    m = prb->m;
    for(ri=prb->m-1; ri>=0; ri--){
        i = prb->rows2use[ri];
		rowidx = i * workvars->n;
        minval = 0;
        minj = -1;
        for(cj=0; cj<prb->n; cj++){
            j = prb->cols2use[cj];
            val = c[rowidx+j] - sol.v[j];
            if(val < minval){
                minval = val;
                minj = j;
            }
        }
        if((minj==-1) || (sol.y[minj]==-1)){
            // this row can be matched without conflicting previous matches
            sol.x[i] = minj;
			if (minj != -1) {
				sol.y[minj] = i;
				C += minval;
			}
            m--;
            prb->rows2use[ri] = prb->rows2use[m];
            prb->rows2use[m] = i;
        }
    }

    for(ri=0; ri<m; ri++){
		i1 = prb->rows2use[ri];
        // shortest paths algorithm
        n = prb->n;
		rowidx = i1 * workvars->n;
        for(cj=0; cj<n; cj++){
            j = prb->cols2use[cj];
            workvars->distances[j] = c[rowidx+j] - sol.v[j];
            workvars->pathback[j] = i1;
        }
		minmissi = i1;
		minmissval = 0;
		for (;;) {
			assert(loopcounter++ < loopescape);
			minval = minmissval;
			minj = -1;
			for (cj = 0; cj < n; cj++) {
				j = prb->cols2use[cj];
				val = workvars->distances[j];
				if (val < minval) {
					minj = j;
					mincol = cj;
					minval = val;
				}
			}
			j = minj;
			if (j == -1) break;
			i = sol.y[j];
			if (i == -1) break;
			// this column should no longer be considered
			sol.v[j] += minval;
			n -= 1;
			prb->cols2use[mincol] = prb->cols2use[n];
			prb->cols2use[n] = j;
			// update distances to other columns
			rowidx = i * workvars->n;
			ui = c[rowidx + j] - sol.v[j];
			if (-ui < minmissval) {
				minmissi = i;
				minmissval = -ui;
			}
			for (cj = 0; cj < n; cj++) {
				j = prb->cols2use[cj];
				val = c[rowidx + j] - sol.v[j] - ui;
				if (val < workvars->distances[j]) {
					workvars->distances[j] = val;
					workvars->pathback[j] = i;
				}
			}
		}
		// travel back through shortest path
		if (j == -1) {
			i = minmissi;
			j = sol.x[i];
			sol.x[i] = -1;
		}
		while (i != i1) {
			assert(loopcounter++ < loopescape);
			i = workvars->pathback[j];
			sol.y[j] = i;
			j1 = j;
			j = sol.x[i];
			sol.x[i] = j1;
		}
		// update reductions
		for (cj = n; cj < prb->n; cj++) {
			sol.v[prb->cols2use[cj]] -= minval;
		}
		// update total cost
		C += minval;
	}
	#ifndef NDEBUG
	    double eps_debug = 0.0000001;
	    for(ri = 0; ri < prb->m; ri++){
            i = prb->rows2use[ri];
		    rowidx = i * workvars->n;
            j = sol.x[i];
            if (j == -1){
			    // check for positive slack on miss row
                for(cj = 0; cj < prb->n; cj++){
				    j1 = prb->cols2use[cj];
				    assert(sol.y[j1] != i);
				    assert(c[rowidx + j1] - sol.v[j1] > -eps_debug);
                }
		    } else {
		        assert(sol.y[j] == i);
                ui = c[rowidx + j] - sol.v[j];
			    assert(ui < eps_debug);
			    // check for positive slack
			    for (cj = 0; cj < prb->n; cj++) {
				    j1 = prb->cols2use[cj];
				    assert(c[rowidx + j1] - sol.v[j1] - ui > -eps_debug);
			    }
		    }
        }
	    for (cj = 0; cj < prb->n; cj++) {
		    j = prb->cols2use[cj];
		    assert(sol.v[j] < eps_debug);
		    if(sol.y[j] == -1){
		        assert(sol.v[j] > -eps_debug);
		    }
	    }
	#endif
	return C;
};

double spStep(double* c, Subproblem* prb, WorkvarsforSSP* workvars, double cost_bound) {
	const double inf = 1000000000;
	int cj, ri, i, i1, j, j1, minj, minmissi, minmissj, mincol, rowidx, n;
	double minval, val, minmissval, ui, missing_cost;
	bool missing, miss_unused, missing_from_row;
	Solution sol = prb->solution;
	#ifndef NDEBUG
	const int loopescape = 10000;
	int loopcounter = 0;
	#endif

	// which row and column are to be rematched
	i1 = prb->rows2use[prb->m-1];
	j1 = sol.x[i1];

	rowidx = i1 * workvars->n;
	// u not necessary to get solution, but gives accurate cost change
	ui = 0;
	if (j1 != -1) {
		ui = c[rowidx + j1] - sol.v[j1];
	}

    n = prb->n;
	for (cj = 0; cj < n; cj++) {
		j = prb->cols2use[cj];
		if(prb->eliminateels[j]){
            workvars->distances[j] = inf;
        } else {
			workvars->pathback[j] = i1;
			workvars->distances[j] = c[rowidx + j] - sol.v[j] - ui;
		}
	}
	minmissj = -1;
	minmissi = i1;
	if (prb->eliminatemiss) minmissval = inf; else minmissval = -ui;
	miss_unused = true;
	missing_from_row = false;
	missing_cost = 0; // this is a dual cost on auxiliary columns
	for (;;) {
		assert(loopcounter++ < loopescape);
		minval = minmissval;
		minj = -2;
		for (cj = 0; cj < n; cj++) {
			j = prb->cols2use[cj];
			val = workvars->distances[j];
			if (val < minval) {
				minj = j;
				minval = val;
				mincol = cj;
			}
		}
		if (minval > cost_bound) return inf; // early stopping
		j = minj;
		if (j == j1) {
			break;
		}
		if (j == -2) {
			if (!miss_unused) {
				//if you got here again, costs must be really high
				return inf;
			}
			// entry to missing zone : row was matched but is now missing
			missing = true;
			missing_from_row = true;
		}
		else {
			i = sol.y[j];
			// this column should no longer be considered
			n -= 1;
			prb->cols2use[mincol] = prb->cols2use[n];
			prb->cols2use[n] = j;
			if (i == -1) {
				// entry to missing zone : col was missing but is now matched
				if (miss_unused) {
					minmissj = j;
					missing = true;
					missing_from_row = false;
				}
				else {
					// already covered the missing zone, this is a dead end
					continue;
				}
			} else {
				missing = false;
			}
		}
		if (missing) {
			if (j1 == -1) {
				j = -1;
				break;
			}
			miss_unused = false;
			missing_cost = minval;
			minmissval = inf;
			ui = -minval;
			// exit from missing zone : row that was missing is matched
			for (ri = 0; ri < prb->m; ri++) {
				i = prb->rows2use[ri];
				if (sol.x[i] == -1) {
					rowidx = i * workvars->n;
					for (cj = 0; cj < n; cj++) {
						j = prb->cols2use[cj];
						val = c[rowidx + j] - sol.v[j] - ui;
						if (val < workvars->distances[j]) {
							workvars->distances[j] = val;
							workvars->pathback[j] = i;
						}
					}
				}
			}
			// exit from missing zone : col that was matched is missing
			for (cj = 0; cj < n; cj++) {
				j = prb->cols2use[cj];
				if (sol.y[j] != -1) {
					val = -sol.v[j] - ui;
					if (val < workvars->distances[j]) {
						workvars->distances[j] = val;
						workvars->pathback[j] = -1;
					}
				}
			}
		}
		else {
			rowidx = i * workvars->n;
			ui = c[rowidx + j] - sol.v[j] - minval;
			if (miss_unused & (-ui < minmissval)) {
				minmissi = i;
				minmissval = -ui;
			}
			for(cj=0; cj<n; cj++){
				j = prb->cols2use[cj];
				val = c[rowidx + j] - sol.v[j] - ui;
				if (val < workvars->distances[j]) {
					workvars->distances[j] = val;
					workvars->pathback[j] = i;
				}
			}
		}
	}
	// augment
	// travel back through shortest path to find matches
	i = i1 + 1; // any number that isn't i1
	while (i != i1) {
		assert(loopcounter++ < loopescape);
		if (j == -1) {
			// exit from missing zone : row was missing but is now matched
			i = -1;
		}
		else {
			i = workvars->pathback[j];
			sol.y[j] = i;
		}
		if (i == -1) {
			// exit from missing zone : column j was matched but is now missing
			if (missing_from_row) {
				// entry to missing zone : row was matched but is now missing
				i = minmissi;
				j = sol.x[i];
				sol.x[i] = -1;
			} else {
				// entry to missing zone : col was missing but is now matched
				j = minmissj;
			}
		} else {
			j1 = j;
			j = sol.x[i];
			sol.x[i] = j1;
		}
	}
	// updating of column reductions
	if (miss_unused) {
		missing_cost = minval;
	}
	for (cj = 0; cj < n; cj++) {
		j = prb->cols2use[cj];
        if (sol.y[j] == -1){
            sol.v[j] = 0;
        } else {
		    sol.v[j] = sol.v[j] + minval - missing_cost;
        }
	}
	for (cj = n; cj < prb->n; cj++) {
		j = prb->cols2use[cj];
		if (sol.y[j] == -1) {
			sol.v[j] = 0;
		}
		else {
			sol.v[j] = sol.v[j] + workvars->distances[j] - missing_cost;
		}
	}
	#ifndef NDEBUG
	    double eps_debug = 0.0000001;
	    for(ri = 0; ri < prb->m; ri++){
            i = prb->rows2use[ri];
		    rowidx = i * workvars->n;
            j = sol.x[i];
            if (j == -1){
			    // check for positive slack on miss row
                for(cj = 0; cj < prb->n; cj++){
				    j1 = prb->cols2use[cj];
				    assert(sol.y[j1] != i);
				    if (!(prb->eliminateels[j1] & (i==i1))){
				        assert(c[rowidx + j1] - sol.v[j1] > -eps_debug);
				    }
                }
		    } else {
		        assert(sol.y[j] == i);
                ui = c[rowidx + j] - sol.v[j];
			    if (!(prb->eliminatemiss & (i==i1))){
			        assert(ui < eps_debug);
			    }
			    // check for positive slack
			    for (cj = 0; cj < prb->n; cj++) {
				    j1 = prb->cols2use[cj];
				    if (!(prb->eliminateels[j1] & (i==i1))){
				        assert(c[rowidx + j1] - sol.v[j1] - ui > -eps_debug);
				    }
			    }
		    }
        }
	    for (cj = 0; cj < prb->n; cj++) {
		    j = prb->cols2use[cj];
		    assert(sol.v[j] < eps_debug);
		    if(sol.y[j] == -1){
		        assert(sol.v[j] > -eps_debug);
		    }
	    }
	#endif
	return minval;
};

#endif
