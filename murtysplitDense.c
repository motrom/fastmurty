/**
Michael Motro github.com/motrom/fastmurty 4/2/19
*/
#ifndef SPARSE

#include "murtysplitDense.h"

WorkvarsforSplit allocateWorkvarsforSplit(int m, int n){
    WorkvarsforSplit workvars;
    workvars.row_best_columns = (int*) malloc((sizeof(double)+sizeof(int))*m);
    workvars.row_cost_estimates = (double*)(workvars.row_best_columns + m);
    workvars.m = m;
    workvars.n = n;
    return workvars;
};

void deallocateWorkvarsforSplit(WorkvarsforSplit workvars){
    free(workvars.row_best_columns);
};

void murtySplit(double* c, Subproblem* prb, WorkvarsforSplit* workvars){
    const double inf = 1000000000;
    int ri, cj, i, j, j2, m2, m3, n2, mincol, maxrow, rowidx;
    int eliminate_i, eliminate_j, deadcol;
    double val, ui, minval, maxval;
    Solution sol;

    sol = prb->solution;
    // put missing columns at beginning
    // they do not need to be fixed for any split
    n2 = 0;
    for(cj=0; cj<prb->n; cj++){
        j = prb->cols2use[cj];
        if(sol.y[j]==-1){
            prb->cols2use[cj] = prb->cols2use[n2];
            prb->cols2use[n2] = j;
            n2++;
        }
    }
    workvars->n_start = n2; // don't split on these columns

    // set aside row m2-1 and its column
    // this row and column had eliminations in the originating problem
    // easiest from the perspective of storing eliminations if this row is eliminated last
    m2 = prb->m - 1; // don't use last row in lookahead
    n2 = prb->n;
    eliminate_j = sol.x[prb->rows2use[m2]];
    if(eliminate_j != -1){
        for(cj=0; cj<n2-1; cj++){
            j = prb->cols2use[cj];
            if(j == eliminate_j){
                prb->cols2use[cj] = prb->cols2use[n2-1];
                prb->cols2use[n2-1] = j;
            }
        }
        n2--; // don't use this column in lookahead
    }

    // determine if all rows will be eliminated or not
    m3 = 0;
    if(workvars->n_start == 0){
        // in this case, you can keep missing rows at the beginning and not fix them
        for(ri=0; ri<m2; ri++){
            i = prb->rows2use[ri];
            if (sol.x[i]==-1){
                prb->rows2use[ri] = prb->rows2use[m3];
                prb->rows2use[m3] = i;
                m3++;
            }
        }
		assert(m3 == m2 - n2);
    }
    workvars->m_start = m3;

    // find estimated cost for row
    // ---> min(c'[i,j] for j!=x[i])
    for(ri=workvars->m_start; ri<m2; ri++){
        i = prb->rows2use[ri];
        rowidx = i*workvars->n;
        j = sol.x[i];
        if (j==-1){
            ui = 0;
            minval = inf;
        } else {
            ui = c[rowidx+j]-sol.v[j];
            minval = 0;
        }
        mincol = -1;
        for(cj = 0; cj < n2; cj++){
            j2 = prb->cols2use[cj];
            if (j2!=j){
                val = c[rowidx+j2] - sol.v[j2];
                if (val < minval){
                    minval = val;
                    mincol = cj;
                }
            }
        }
        workvars->row_cost_estimates[ri] = minval - ui;
        workvars->row_best_columns[ri] = mincol;
        //workvars->row_cost_estimates[ri] += .00001*ri; // just for debugging!! so that order is always the same
    }

    for(m3=m2-1; m3>=workvars->m_start; m3--){
        // choose the worst current row and partition on it last
        // meaning that subproblem has the fewest fixed rows --> the biggest size
        maxrow = -1;
        maxval = -1;
        for(ri=workvars->m_start; ri<=m3; ri++){
            if(workvars->row_cost_estimates[ri] > maxval){
                maxrow = ri;
                maxval = workvars->row_cost_estimates[ri];
            }
        }
		assert(maxrow >= 0);
        eliminate_i = prb->rows2use[maxrow];
        prb->rows2use[maxrow] = prb->rows2use[m3];
        prb->rows2use[m3] = eliminate_i;
        // don't want to pick this row again, overwrite it in workvars
        workvars->row_cost_estimates[maxrow] = workvars->row_cost_estimates[m3];
        workvars->row_best_columns[maxrow] = workvars->row_best_columns[m3];

        eliminate_j = sol.x[eliminate_i];
        if(eliminate_j != -1){
            // swap columns so this particular column matches that row
            deadcol = -2;
            for(cj=0; cj<n2; cj++){
                if (prb->cols2use[cj] == eliminate_j){
                    deadcol = cj;
                    n2--;
                    prb->cols2use[cj] = prb->cols2use[n2];
                    prb->cols2use[n2] = eliminate_j;
                    break;
                }
            }
            // update other cost estimates that had picked the same column
            for(ri=workvars->m_start; ri<m3; ri++){
                if (workvars->row_best_columns[ri] == deadcol){
                    // recalculate lookahead bound without eliminate_j
                    i = prb->rows2use[ri];
                    rowidx = i*workvars->n;
                    j = sol.x[i];
                    if (j==-1){
                        ui = 0;
                        minval = inf;
                    } else {
                        ui = c[rowidx+j]-sol.v[j];
                        minval = 0;
                    }
                    mincol = -1;
                    for(cj = 0; cj < n2; cj++){
                        j2 = prb->cols2use[cj];
                        if (j2!=j){
                            val = c[rowidx+j2] - sol.v[j2];
                            if (val < minval){
                                minval = val;
                                mincol = cj;
                            }
                        }
                    }
                    workvars->row_cost_estimates[ri] = minval - ui;
                    workvars->row_best_columns[ri] = mincol;
                    //workvars->row_cost_estimates[ri] += .00001*ri;// just for debugging!! so that order is always the same
                }
            }
        }
    }
};

#endif
