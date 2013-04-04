#include "rapidxml.hpp"
#include "rapidxml_utils.hpp"
#include "algorithm.cuh"
#include <algorithm>


void check(cudaError call){
	if(call != cudaSuccess){
		throw &call;
	}
}

int chromosomeCheck(int chromosome[], geneticAlgorithm * hostAlgorithm){
	int k;
	for(int i = 0; i < hostAlgorithm->CHROMOSOME_SIZE; i++){
		k = 0;
		for(int j = 0; j < hostAlgorithm->CHROMOSOME_SIZE; j++){
			if(chromosome[j] == i){
				k++;
			}else if(chromosome[j] > hostAlgorithm->CHROMOSOME_SIZE || chromosome[j] < 0){
				return 1;
			}
		}
		if(k != 1){
			return 1;
		}
	}
	return 0;
}

void readDataFromXMLInstance(rapidxml::xml_node<>* graph, geneticAlgorithm * hostAlgorithm){
	rapidxml::xml_node<>* vertex = graph->first_node("vertex");
	rapidxml::xml_node<>* edge = vertex->first_node("edge");
	rapidxml::xml_attribute<>* cost;
	for(int i = 0; i < hostAlgorithm->CHROMOSOME_SIZE; i++){
		hostAlgorithm->adjacencyMatrix[i*hostAlgorithm->CHROMOSOME_SIZE+i] = 0.0;
	}

	for(int i = 0; i < hostAlgorithm->CHROMOSOME_SIZE; i++){
		while(true){
			cost = edge->first_attribute("cost");
			hostAlgorithm->adjacencyMatrix[i*(hostAlgorithm->CHROMOSOME_SIZE)+atoi(edge->value())] = atof(cost->value());
			if(edge != vertex->last_node("edge")){
				edge = edge->next_sibling("edge");
			}else{
				break;
			}
		}

		if(vertex != graph->last_node("vertex")){
			vertex = vertex->next_sibling("vertex");
			edge = vertex->first_node("edge");
		}
	}
}

void runGeneticAlgorithm(geneticAlgorithm * deviceAlgorithm){
	createRandomPermutation<<<GRID_SIZE, BLOCK_SIZE>>>(*deviceAlgorithm);
	createRandomSeeds<<<GRID_SIZE, BLOCK_SIZE>>>(*deviceAlgorithm, time(NULL));
	check(cudaDeviceSynchronize());

	bool stop = false;


	for(int i = 0; i < deviceAlgorithm->GENERATIONS; i++){
		runOneGeneration<<<GRID_SIZE, BLOCK_SIZE>>>(*deviceAlgorithm);
		check(cudaDeviceSynchronize());
		runOneMigration<<<GRID_SIZE, BLOCK_SIZE>>>(*deviceAlgorithm);
		check(cudaDeviceSynchronize());
		for(int i = 0; i < GRID_SIZE; i++){
			if(deviceAlgorithm->optimalLengthReached[i]){
				stop = true;
				break;
			}
		}
		if(stop) break;
		deviceAlgorithm->runGenerations++;
	}
}

int main(int argc, char ** argv){
	try{
		char* filename = argv[1];
		int optimalLength = atoi(argv[2]);
		int generations = atoi(argv[3]);


		rapidxml::xml_document<> doc;
		rapidxml::file<> file(filename);
		doc.parse<0>(file.data());
		rapidxml::xml_node<>* graph = doc.first_node("travellingSalesmanProblemInstance")->first_node("graph");


		geneticAlgorithm * hostAlgorithm = new geneticAlgorithm(generations, optimalLength, rapidxml::count_children(graph));
		geneticAlgorithm * deviceAlgorithm = new geneticAlgorithm(generations, optimalLength, rapidxml::count_children(graph));

		cudaDeviceSetLimit(cudaLimitMallocHeapSize, 33554432);

		(hostAlgorithm->seeds) = new int[hostAlgorithm->POPULATION_SIZE];
		check(cudaMalloc((void**) &(deviceAlgorithm->seeds), hostAlgorithm->POPULATION_SIZE*sizeof(int)));

		(hostAlgorithm->source) = new int[hostAlgorithm->CHROMOSOME_SIZE];
		check(cudaMalloc((void**) &(deviceAlgorithm->source), hostAlgorithm->CHROMOSOME_SIZE*sizeof(int)));

		(hostAlgorithm->adjacencyMatrix) = new double[hostAlgorithm->CHROMOSOME_SIZE*hostAlgorithm->CHROMOSOME_SIZE];
		check(cudaMalloc((void**) &(deviceAlgorithm->adjacencyMatrix), hostAlgorithm->CHROMOSOME_SIZE*hostAlgorithm->CHROMOSOME_SIZE*sizeof(double)));

		(hostAlgorithm->populationChromosome) = new int[hostAlgorithm->POPULATION_SIZE*hostAlgorithm->CHROMOSOME_SIZE];
		check(cudaMalloc((void**) &(deviceAlgorithm->populationChromosome), sizeof(int)*hostAlgorithm->POPULATION_SIZE*hostAlgorithm->CHROMOSOME_SIZE));

		(hostAlgorithm->populationDistance) = new double[hostAlgorithm->POPULATION_SIZE];
		check(cudaMalloc((void**) &(deviceAlgorithm->populationDistance), sizeof(double)*hostAlgorithm->POPULATION_SIZE));

		readDataFromXMLInstance(graph, hostAlgorithm);
		check(cudaMemcpy(deviceAlgorithm->adjacencyMatrix, hostAlgorithm->adjacencyMatrix, sizeof(double)*hostAlgorithm->CHROMOSOME_SIZE*hostAlgorithm->CHROMOSOME_SIZE, cudaMemcpyHostToDevice));


/*		for(int i = 0; i < hostAlgorithm->CHROMOSOME_SIZE; i++){
			for(int j = 0; j < hostAlgorithm->CHROMOSOME_SIZE; j++){
				std::cout << hostAlgorithm->adjacencyMatrix[i*hostAlgorithm->CHROMOSOME_SIZE+j] << " ";
			}
			std::cout << std::endl;
		}
*/

		for(int i = 0; i < hostAlgorithm->CHROMOSOME_SIZE; i++){
			hostAlgorithm->source[i] = i;
		}

		check(cudaMemcpy(deviceAlgorithm->source, hostAlgorithm->source, sizeof(int)*hostAlgorithm->CHROMOSOME_SIZE, cudaMemcpyHostToDevice));


		runGeneticAlgorithm(deviceAlgorithm);

		check(cudaMemcpy(hostAlgorithm->populationDistance ,deviceAlgorithm->populationDistance, sizeof(double)*hostAlgorithm->POPULATION_SIZE, cudaMemcpyDeviceToHost));
		check(cudaMemcpy(hostAlgorithm->populationChromosome ,deviceAlgorithm->populationChromosome, sizeof(int)*hostAlgorithm->POPULATION_SIZE*hostAlgorithm->CHROMOSOME_SIZE, cudaMemcpyDeviceToHost));

		double bestDistances[GRID_SIZE*4];
		for(int i = 0; i < GRID_SIZE*4; i++){
			bestDistances[i] = hostAlgorithm->populationDistance[i*((hostAlgorithm->ISLAND_POPULATION_SIZE)/4)];
		}


/*		for (int i = 0; i < hostAlgorithm->POPULATION_SIZE; i++){
			std::cout << '[' << chromosomeCheck(&(hostAlgorithm->populationChromosome[i*hostAlgorithm->CHROMOSOME_SIZE]), hostAlgorithm) << ']' << " ";
			for(int j = 0; j < hostAlgorithm->CHROMOSOME_SIZE; j++){
				std::cout << hostAlgorithm->populationChromosome[i*hostAlgorithm->CHROMOSOME_SIZE+j] << " ";
			}
			std::cout << hostAlgorithm->populationDistance[i] << std::endl;
		}
*/


		std::sort(bestDistances, &bestDistances[GRID_SIZE]);

		for(int i = 0; i < GRID_SIZE; i++){
			std::cout << bestDistances[i] << std::endl; 
		}

		std::cout << deviceAlgorithm->runGenerations << " ";


		bool allGood = true;
		for(int i = 0; i < hostAlgorithm->POPULATION_SIZE; i++){
			if(chromosomeCheck(&(hostAlgorithm->populationChromosome[i*hostAlgorithm->CHROMOSOME_SIZE]), hostAlgorithm) == 1){
				allGood = false;
				break;
			}
		}

		if(allGood){
			std::cout << "All good Captian" << std::endl;
		}else{
			std::cout << "Well Hitlerfuck" << std::endl;
		}

		delete hostAlgorithm->seeds;
		delete hostAlgorithm->source;
		delete hostAlgorithm->adjacencyMatrix;
		delete hostAlgorithm->populationChromosome;
		delete hostAlgorithm->populationDistance;
		delete hostAlgorithm;
		cudaFree(deviceAlgorithm->seeds);
		cudaFree(deviceAlgorithm->source);
		cudaFree(deviceAlgorithm->adjacencyMatrix);
		cudaFree(deviceAlgorithm->populationChromosome);
		cudaFree(deviceAlgorithm->populationDistance);
		delete deviceAlgorithm;

	}
	catch(cudaError * e){
		std::cout << "Oh crap: " << cudaGetErrorString(*e) << std::endl;
	}
}



