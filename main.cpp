#include <TSP.hpp>

int main(int argc, char** argv){
	TSP TSPGraph (argv[0]);
	for (double* i: TSPGraph.list() ){
		for (double j : i){
			std::cout << j << ' ';
		}
		std::cout << '\n';
	}
}
