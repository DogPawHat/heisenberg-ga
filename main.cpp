#include "TSP.hpp"
#include <iostream>
#include <istream>
#include <fstream>

int main(int argc, char** argv){
	char q;
	TSP TSPGraph (argv[1]);
	for (std::vector<double> i: TSPGraph.list()){
		std::cout << i[0] << ' ' << i[1] << std::endl;
	}
	std::cout << "Press q to quit";
	std::cin >> q;
}
