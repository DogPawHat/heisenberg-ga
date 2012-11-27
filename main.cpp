<<<<<<< HEAD

=======
>>>>>>> Fixing things
#include "TSP.h"
#include <vector>
#include <iostream>
#include <fstream>

<<<<<<< HEAD
int main(int argc, char** argv){

=======
int main(int argc, char** argv){
>>>>>>> Fixing things
	TSP TSPGraph (argv[1]);
	char l;
	std::vector< std::vector<double> > list = TSPGraph.TSP::list();
	for (unsigned int i = 0; i < list.size(); i++){
		std::cout << list[i][0] << ' ' << list[i][1] << std::endl;
	}
	std::cout << "Press any key to exit";
	std::cin >> l;
}
