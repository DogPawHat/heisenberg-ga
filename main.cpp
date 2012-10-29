#include "TSP.hpp"
#include <stdio.h>
#include <iostream>

int main(int argc, char** argv){
	TSP TSPGraph (argv[0]);
	for (double *i: TSPGraph.list()){
		std::cout << i[0] << ' ' << i[1] << '\n';
	}
}
