#ifndef TSPLIST_H
#define TSPLIST_H

#include <vector>

typedef std::vector< std::vector<double> > TSPList ;

double** TurnTSPListTo2DArray(TSPList list){
	double** result;
	result = new double*[list.size()];

	for(int i = 0; i < list.size(); i++){
		result[i] = new double[2];
		for(int j = 0; j < 2; j++){
			result[i][j] = list[i][j];
		}
	}

	return result;
}

#endif
