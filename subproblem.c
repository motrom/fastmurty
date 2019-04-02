/**
Michael Motro github.com/motrom/fastmurty 4/2/19
*/
#include "subproblem.h"
#include <string.h> // memcpy

Subproblem allocateSubproblem(int m, int n, char *buffer) {
	Subproblem prb;
	prb.m = m;
	prb.n = n;
	prb.buffer = buffer;
	prb.solution.x = (int*) buffer;
	prb.solution.y = (int*) (prb.solution.x + m);
	prb.solution.v = (double*) (prb.solution.y + n);
	prb.rows2use = (int*) (prb.solution.v + n);
	prb.cols2use = (int*) (prb.rows2use + m);
	prb.eliminateels = (bool*) (prb.cols2use + n);
	prb.eliminatemiss = false;
	return prb;
};

Subproblem createSubproblem(int m, int n) {
	size_t subproblemsize = sizeof(int)*(m + n) * 2;
	subproblemsize += sizeof(bool)*n + sizeof(double)*n;
	char* buffer = (char*) malloc(subproblemsize);
	Subproblem prb = allocateSubproblem(m, n, buffer);
	return prb;
};

void deallocateSubproblem(Subproblem* prb) {
	free(prb->buffer);
};

size_t sizeofSolutionData(int m, int n){
	return sizeof(int)*m + sizeof(int)*n + sizeof(double)*n;
};

size_t sizeofSubproblemData(int m, int n) {
	return sizeofSolutionData(m, n) + sizeof(int)*(m + n) + sizeof(bool)*n;
};

void copySubproblem(Subproblem* outprb, Subproblem* inprb, size_t size) {
	memcpy(outprb->buffer, inprb->buffer, size);
	outprb->m = inprb->m;
	outprb->n = inprb->n;
	outprb->eliminatemiss = inprb->eliminatemiss;
};

void copySolution(Subproblem* outprb, Subproblem* inprb, size_t size) {
	memcpy(outprb->buffer, inprb->buffer, size);
};
