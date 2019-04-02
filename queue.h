/**
Michael Motro github.com/motrom/fastmurty 4/2/19
*/
#include "subproblem.h"

typedef struct QueueEntryStruct {
	double key;
	Subproblem* val;
} QueueEntry;

QueueEntry qPopMin(QueueEntry* Q, int Qsize);
QueueEntry qReplaceMax(QueueEntry* Q, QueueEntry newele, int Qsize);
