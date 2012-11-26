#include "TSP.h"
#include <iostream>
#include <fstream>

int main(int argc, char** argv){
	TSP TSPGraph (argv[1]);
	for (std::vector<double> i: TSPGraph.list()){
		std::cout << i[0] << ' ' << i[1] << std::endl;
	}
}
