
#include "TSP.h"
#include <vector>
#include <iostream>
#include <fstream>

int main(int argc, char** argv){

	TSP TSPGraph (argv[1]);
	char l;
	std::vector< std::vector<double> > list = TSPGraph.TSP::list();
	for (unsigned int i = 0; i < list.size(); i++){
		std::cout << list[i][0] << ' ' << list[i][1] << std::endl;
	}
	std::cout << "Press any key to exit";
	std::cin >> l;
}
