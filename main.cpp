#include "parser/TSP.h"
#include "TSPList.h"
#include <iostream>
#include <fstream>

int main(int argc, char** argv){

	TSP TSPGraph (argv[1]);
	TSPList list = TSPGraph.list();
	for (int i = 0; i < list.size(); i++){
		std::cout << list[i][0] << ' ' << list[i][1] << std::endl;
	}
}
