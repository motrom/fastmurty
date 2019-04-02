/**
Michael Motro github.com/motrom/fastmurty 4/2/19
functionality - double ended priority queue
implementation - interval heap
based on code from "A Comparative Analysis of Three Different Priority Deques", Skov and Olsen
*/
#include "queue.h"

QueueEntry minele, newele, hichild, lochild;
double newkey, childkey, rightkey;
int pos, childpos, rightpos, size_limit, cornerstate;

/* remove smallest entry, decreasing size
   assumes length of Q before pop is Qsize+1,
    length after pop is Qsize
*/
QueueEntry qPopMin(QueueEntry* Q, int Qsize) {
	newele = Q[Qsize];
	// put minimum in back
	// queue ends up as a reverse-ordered list of the best solutions
	// not necessary but maybe useful?
    Q[Qsize] = Q[0];
	newkey = newele.key;
	pos = 0;
	childpos = 2; // leftmost child position, lo index
	while (childpos < Qsize) {
		childkey = Q[childpos].key;
		// Set childpos to index of smaller child.
		rightpos = childpos + 2;
		if(rightpos < Qsize){
		    rightkey = Q[rightpos].key;
			if(rightkey < childkey){
				// branch to the right instead of the left
				childpos = rightpos;
				childkey = rightkey;
            }
        }
		if (newkey < childkey){
		    break; // the new element is correctly positioned at pos
        }
		// move the smaller child up.
		Q[pos] = Q[childpos];
		// swap low and high if needed
        rightpos = childpos+1;
        if(rightpos < Qsize){
		    hichild = Q[rightpos];
		    if (newkey > hichild.key){
                Q[rightpos] = newele;
                newele = hichild;
                newkey = newele.key;
            }
        }
		pos = childpos;
		childpos = (pos << 1) + 2;
	}
    Q[pos] = newele;
	return Q[Qsize];
};


/* replace max element with newele, reorder the heap and return the new max element
   assumes Qsize >= 1, and that newele is less than the max element (not checked here!)
*/
QueueEntry qReplaceMax(QueueEntry* Q, QueueEntry newele, int Qsize){
    if (Qsize == 1){
        Q[0] = newele;
        return newele;
    }
    pos = 1;
    childpos = 3; // leftmost child position, lo index
    //newele = Q[1];
    newkey = newele.key; 
    lochild = Q[0];
    if (newkey < lochild.key){
        // new element is smallest in heap
        Q[0] = newele;
        newele = lochild;
        newkey = newele.key;
    }
    size_limit = (Qsize-2) & -4;
    while (childpos < size_limit){
        childkey = Q[childpos].key;
        // set childpos to index of larger child
        rightpos = childpos + 2;
        rightkey = Q[rightpos].key;
        if (rightkey > childkey){
            // branch to the right instead of left
            childpos = rightpos;
			childkey = rightkey;
        }
        if (newkey > childkey){
            // the new element is correctly positioned at pos
            break;        
        }
        // move the smaller child up
        Q[pos] = Q[childpos];
        // swap low and high if needed
        lochild = Q[childpos-1];
        if (newkey < lochild.key){
            Q[childpos-1] = newele;
            newele = lochild;
            newkey = newele.key;
        }
        pos = childpos;
        childpos = (pos << 1) + 1;
    }
    // address corner case at end of heap
    cornerstate = Qsize - childpos;
    if (cornerstate >= 0){
        // if cornerstate 0,
        if(cornerstate == 0){
            // only lo index of left child is present
            childpos -= 1;        
        } else if (cornerstate == 2){
            // lo index of right child present
            rightpos = childpos+1;
            if (Q[rightpos].key > Q[childpos].key){
                childpos = rightpos;
            }
        }
        if (newkey < Q[childpos].key){
            // move down once more
            Q[pos] = Q[childpos];
            pos = childpos;
            if(pos & 1){
                // picked hi index of left child, check for swap
                lochild = Q[pos-1];
                if(newkey < lochild.key){
                    Q[pos-1] = newele;
                    newele = lochild;
                    newkey = newele.key;
                }
            }
        }
    }
    Q[pos] = newele;
    return Q[1];
};
