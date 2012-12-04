#include "TSP.h"
#include <vector>
#include <iostream>
#include <fstream>
#include "TSPList.h"



int main(int argc, char** argv){
	TSP TSPGraph (argv[1]);
	TSPList list = TSPGraph.list();
	double** test2D = TurnTSPListTo2DArray(list);

	for (unsigned int i = 0; i < list.size(); i++){
		std::cout << test2D[i][0] << ' ' << test2D[i][1] << std::endl;
	}
	std::cout << "Press any key to exit";
	std::cin >> new char;
	delete test2D;
}
