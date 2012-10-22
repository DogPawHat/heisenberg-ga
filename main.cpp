#include "TSP.h"
#include <ostream>

int main(int argc, char** argv){
	TSP *TSPGraph = new TSP(argv[0]);
	for each(float* i in TSP::list){
		for each(float j in i){
			std::cout << i << ' ';
		}
		std::cout << '/n';
	}
}