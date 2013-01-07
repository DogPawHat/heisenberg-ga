#ifndef TSPLIST_H
#define TSPLIST_H

#include <vector>

typedef std::vector< std::vector<double> > TSPList ;

double** TurnTSPListTo2DArray(TSPList list);

#endif
