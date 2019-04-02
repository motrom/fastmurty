/**
Michael Motro github.com/motrom/fastmurty 4/2/19
*/
#ifdef SPARSE

#include "sspSparse.h"
#include <stddef.h> // size_t

WorkvarsforSSP allocateWorkvarsforSSP(int m, int n) {
	WorkvarsforSSP workvars;
    size_t totalsize = sizeof(Pathtype) * (m*n+m) +
                       sizeof(int) * n;
	workvars.Q = (Pathtype *) malloc(totalsize);
	workvars.pathback = (int *)(workvars.Q + m*n+m);
	workvars.m = m;
	workvars.n = n;
	return workvars;
};

void deallocateWorkvarsforSSP(WorkvarsforSSP workvars) {
	free(workvars.Q);
};


/* qPush and qPop are a standard binary heap push and pop
    unlike the interval heap of queue.c */
void qPush(Pathtype* Q, int pos, double newval, int newi, int newj){
    int parentpos;
    while (pos > 0){
        parentpos = (pos - 1) >> 1;
        if (newval > Q[parentpos].val){
            break;
        }
        Q[pos] = Q[parentpos];
        pos = parentpos;
    }
    Q[pos].val = newval;
    Q[pos].i = newi;
    Q[pos].j = newj;
};

Pathtype qPop(Pathtype* Q, int Qsize){
    Pathtype minele, newele;
    double newkey;
    int pos, childpos, rightpos;
    minele = Q[0];
    newele = Q[Qsize];
    newkey = newele.val;
    pos = 0;
    childpos = 1;
    while (childpos < Qsize){
        rightpos = childpos + 1;
        if (rightpos < Qsize){
            if (Q[childpos].val > Q[rightpos].val){
                childpos = rightpos;
            }
        }
        if (Q[childpos].val > newkey){
            break;
        }
        Q[pos] = Q[childpos];
        pos = childpos;
        childpos = (pos << 1) + 1;
    }
    Q[pos] = newele;
    return minele;
};

double SSP(cs_di c, Subproblem* prb, WorkvarsforSSP* workvars){
    const double inf = 1000000000;
	double C = 0;
	int cj, ri, i, i1, j, j1, minj, m, cstartidx, cendidx, Qsize;
    double minval, val, ui;
	Solution sol = prb->solution;
	Pathtype minpath;
	Pathtype* Q;
	int* pathback;
	#ifndef NDEBUG
	const int loopescape = 10000;
	int loopcounter = 0;
	#endif
	
	ui = 0;
	
	pathback = workvars->pathback;
	Q = workvars->Q;

    // reset y, v, pathback
    // cols2use indexing and the row-sparse matrix don't play well together
    // so inclusion information is handled by pathback
    for(j = 0; j < c.n; j++){
        pathback[j] = 0;
    }
    for(cj = 0; cj < prb->n; cj++){
        j = prb->cols2use[cj];
        sol.y[j] = -1;
        sol.v[j] = 0;
        pathback[j] = -2;
    }


    // basic column reduction - basically running some rows in a convenient order
    m = prb->m;
    for(ri=prb->m-1; ri>=0; ri--){
        i = prb->rows2use[ri];
        minval = 0;
        minj = -1;
        cstartidx = c.p[i];
        cendidx = c.p[i+1];
        for(cj=cstartidx; cj<cendidx; cj++){
            j = c.i[cj];
            if(pathback[j] == -2){
                val = c.x[cj];
                if(val < minval){
                    minval = val;
                    minj = j;
                }
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
		Qsize = 0; // restart queue
		for(cj = 0; cj < prb->n; cj++){
            pathback[prb->cols2use[cj]] = -2;
        }
        // shortest paths algorithm
		cstartidx = c.p[i1];
		cendidx = c.p[i1+1];
        for(cj=cstartidx; cj<cendidx; cj++){
            j = c.i[cj];
            if (pathback[j] == -2){
                qPush(Q, Qsize, c.x[cj] - sol.v[j], i1, j);
                Qsize++;
            }
        }
        qPush(Q, Qsize, 0, i1, -1);
        Qsize++;
		for (;;) {
			for (;;) {
				assert(loopcounter++ < loopescape);
			    if (Qsize == 0){
			        return inf;
			    }
			    Qsize--;
			    minpath = qPop(Q, Qsize);
			    j = minpath.j;
			    if (j == -1) break; // hit unmatched row
	            if (pathback[j] == -2) break;
			}
			i = minpath.i;
			minval = minpath.val;
			if (j == -1) break;
			// this column should no longer be considered
			pathback[j] = i;
			sol.v[j] += minval;
			i = sol.y[j];
			if (i == -1) break;
			// update distances to other columns
			cstartidx = c.p[i];
			cendidx = c.p[i+1];
			// have to look up the right column for ui
			for (cj=cstartidx; cj<cendidx; cj++){
			    if (c.i[cj] == j){
			        ui = c.x[cj] - sol.v[j];
			    }
			}
			qPush(Q, Qsize, -ui, i, -1);
			Qsize++;
			for (cj=cstartidx; cj<cendidx; cj++){
				j = c.i[cj];
				if (pathback[j] == -2){
				    val = c.x[cj] - sol.v[j] - ui;
				    qPush(Q, Qsize, val, i, j);
				    Qsize++;
				}
			}
		}
		// travel back through shortest path
		if (j == -1) {
			j = sol.x[i];
			sol.x[i] = -1;
		}
		while (i != i1) {
			assert(loopcounter++ < loopescape);
			i = pathback[j];
			sol.y[j] = i;
			j1 = j;
			j = sol.x[i];
			sol.x[i] = j1;
		}
		// update reductions
		for (cj = 0; cj < prb->n; cj++) {
		    j = prb->cols2use[cj];
		    if (pathback[j] != -2){
			    sol.v[j] -= minval;
			}
		}
        
		// update total cost
		C += minval;
	}
	#ifndef NDEBUG
	double eps_debug = 0.0000001;
	for(j=0; j<workvars->n; j++){
	    pathback[j] = 0;
	}
	for(cj=0; cj<prb->n; cj++){
	    j = prb->cols2use[cj];
	    pathback[j] = 1;
	}
	for (ri=0; ri<prb->m; ri++){
	    i = prb->rows2use[ri];
	    j = sol.x[i];
	    cstartidx = c.p[i];
	    cendidx = c.p[i+1];
	    if (j==-1){
		    // check for positive slack on miss row
            for(cj = cstartidx; cj < cendidx; cj++){
                j1 = c.i[cj];
			    assert(sol.y[j1] != i);
			    if (pathback[j1]){
			        assert(c.x[cj] - sol.v[j1] > -eps_debug);
			    }
            }
	    } else {
	        // check for negative reduction and positive slack on match row
		    assert(sol.y[j] == i);
		    for(cj=cstartidx; cj<cendidx; cj++){
		        j1 = c.i[cj];
		        if (c.i[cj] == j){
		            ui = c.x[cj] - sol.v[j];
		        }
		    }
		    assert(ui < eps_debug);
		    for (cj=cstartidx; cj<cendidx; cj++) {
		        j1 = c.i[cj];
			    if ((j1 != j) & pathback[j1]){
			        assert(c.x[cj] - sol.v[j1] - ui > -eps_debug);
			    }
		    }
	    }
	}
	#endif
	return C;
};

double spStep(cs_di c, Subproblem* prb, WorkvarsforSSP* workvars, double cost_bound) {
	const double inf = 1000000000;
	int cj, ri, Qsize, i, i1, j, j1, cstartidx, cendidx;
	int minmissi = 0;//-1;
	int minmissj = 0;//i1;
	double minval, val, ui, missing_cost;
	bool missing, miss_unused, missing_from_row;
	Solution sol = prb->solution;
	Pathtype minpath;
	Pathtype* Q;
	int* pathback;
	#ifndef NDEBUG
    const int loopescape = 1000;
    int loopcounter = 0;
	#endif

    // cols2use indexing the row-sparse matrix don't play well together
    // so inclusion information is handled by pathback
    Q = workvars->Q;
    Qsize = 0;
    pathback = workvars->pathback;
    for(j = 0; j < c.n; j++){
        pathback[j] = 0;
    }
    for(cj = 0; cj < prb->n; cj++){
        pathback[prb->cols2use[cj]] = -2;
    }

	// which row and column are to be rematched
	i1 = prb->rows2use[prb->m-1];
	j1 = sol.x[i1];

	cstartidx = c.p[i1];
	cendidx = c.p[i1+1];
	// u not necessary to get solution, but gives accurate cost change
	ui = 0;
	for (cj=cstartidx; cj<cendidx; cj++){
	    if (c.i[cj] == j1){
	        ui = c.x[cj] - sol.v[j1];
	    }
	}
    for(cj=cstartidx; cj<cendidx; cj++){
        j = c.i[cj];
        if ((pathback[j] == -2) & (!prb->eliminateels[j])){
            val = c.x[cj] - sol.v[j] - ui;
            if (val < cost_bound){
                qPush(Q, Qsize, val, i1, j);
                Qsize++;
            }
        }
    }
    if (!prb->eliminatemiss){
        val = -ui;
        if (val < cost_bound){
            qPush(Q, Qsize, val, i1, -1);
            Qsize++;
        }
    }
	miss_unused = true;
	missing_from_row = false;
	missing_cost = 0; // this is a dual cost on auxiliary columns
	for (;;) {
		missing = false;
		for (;;) {
			assert(loopcounter++ < loopescape);
		    if (Qsize == 0){
		        return inf; // early stopping
		    }
		    Qsize--;
		    minpath = qPop(Q, Qsize);
		    if (minpath.j == -1){
		        if (miss_unused){
		            minmissi = minpath.i;
		            missing = true;
		            missing_from_row = true;
		            break; // hit unmatched row
		        }
            }
            if (pathback[minpath.j] == -2) break;
		}
		minval = minpath.val;
		if (!missing){
		    i = minpath.i;
		    j = minpath.j;
		    // this column should no longer be considered
		    pathback[j] = i;
		    sol.v[j] += minval;
		    if (j == j1) {
			    break;
		    }
		    i = sol.y[j];
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
			}
		}
		if (missing) {
			if (j1 == -1) {
				j = -1;
				break;
			}
			miss_unused = false;
			missing_cost = minval;
			ui = -minval;
			// exit from missing zone : row that was missing is matched
			for (ri = 0; ri < prb->m; ri++) {
				i = prb->rows2use[ri];
				if (sol.x[i] == -1) {
					cstartidx = c.p[i];
					cendidx = c.p[i+1];
					for (cj = cstartidx; cj < cendidx; cj++) {
						j = c.i[cj];
						if (pathback[j]==-2){
						    val = c.x[cj] - sol.v[j] - ui;
						    if (val < cost_bound) {
							    qPush(Q, Qsize, val, i, j);
							    Qsize++;
						    }
						}
					}
				}
			}
			// exit from missing zone : col that was matched is missing
			for (cj = 0; cj < prb->n; cj++) {
				j = prb->cols2use[cj];
				if ((sol.y[j] != -1) & (pathback[j]==-2)) {
					val = -sol.v[j] - ui;
					if (val < cost_bound) {
						qPush(Q, Qsize, val, -1, j);
					    Qsize++;
					}
				}
			}
		}
		else {
		    // update distances to other columns
		    cstartidx = c.p[i];
			cendidx = c.p[i+1];
			// have to look up the right column for ui
			for (cj=cstartidx; cj<cendidx; cj++){
			    if (c.i[cj] == j){
			        ui = c.x[cj] - sol.v[j];
			    }
			}
			if (miss_unused){
			    val = -ui;
			    if (val < cost_bound){
			        qPush(Q, Qsize, -ui, i, -1);
			        Qsize++;
			    }
			}
			for (cj = cstartidx; cj < cendidx; cj++) {
				j = c.i[cj];
				if (pathback[j]==-2){
				    val = c.x[cj] - sol.v[j] - ui;
				    if (val < cost_bound) {
					    qPush(Q, Qsize, val, i, j);
					    Qsize++;
				    }
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
		} else {
			i = pathback[j];
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
	double updatevalchosen, updatevalnot;
	if (miss_unused) {
	    updatevalchosen = -minval;
	} else {
	    updatevalchosen = -missing_cost;
	}
	updatevalnot = minval + updatevalchosen;
	
	for (cj = 0; cj < prb->n; cj++) {
		j = prb->cols2use[cj];
        if (sol.y[j] == -1){
            sol.v[j] = 0;
        } else {
            if (pathback[j] == -2){
		        sol.v[j] += updatevalnot;
		    } else {
		        sol.v[j] += updatevalchosen;
		    }
        }
	}
	#ifndef NDEBUG
	    double eps_debug = 0.0000001;
	    bool eliminatingrow;
	    for(j=0; j<c.n; j++){
	        pathback[j] = 0;
	    }
	    for(cj=0; cj<prb->n; cj++){
	        j = prb->cols2use[cj];
	        pathback[j] = 1;
	    }
	    for (ri=0; ri<prb->m; ri++){
	        i = prb->rows2use[ri];
	        eliminatingrow = (i == i1);
	        j = sol.x[i];
	        cstartidx = c.p[i];
	        cendidx = c.p[i+1];
	        if (j==-1){
		        // check for positive slack on miss row
                for(cj = cstartidx; cj < cendidx; cj++){
                    j1 = c.i[cj];
			        assert(sol.y[j1] != i);
			        if (pathback[j1] & (!(eliminatingrow & prb->eliminateels[j1]))){
			            assert(c.x[cj] - sol.v[j1] > -eps_debug);
			        }
                }
	        } else {
	            // check for negative reduction and positive slack on match row
		        assert(sol.y[j] == i);
		        for(cj=cstartidx; cj<cendidx; cj++){
		            if (c.i[cj] == j){
		                ui = c.x[cj] - sol.v[j];
		            }
		        }
		        if(!(eliminatingrow & prb->eliminatemiss)){
		            assert(ui < eps_debug);
		        }
		        for (cj=cstartidx; cj<cendidx; cj++) {
		            j1 = c.i[cj];
			        if ((j1 != j) & pathback[j1] & (!(eliminatingrow & prb->eliminateels[j1]))){
			            assert(c.x[cj] - sol.v[j1] - ui > -eps_debug);
			        }
		        }
	        }
	    }
	#endif
	return minval;
};

#endif
