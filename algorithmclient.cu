#include "rapidxml.hpp"
#include "rapidxml_utils.hpp"
#include "algorithm.cuh"


geneticAlgorithm * hostAlgorithm;
geneticAlgorithm * deviceAlgorithm;


void check(cudaError call){
	if(call != cudaSuccess){
		throw &call;
	}
}

int chromosomeCheck(int chromosome[]){
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

void readDataFromXMLInstance(rapidxml::xml_node<>* graph){
	rapidxml::xml_node<>* vertex = graph->first_node("vertex");
	rapidxml::xml_node<>* edge;
	rapidxml::xml_attribute<>* cost;
	for(int i = 0; i < hostAlgorithm->CHROMOSOME_SIZE; i++){
		edge = vertex->first_node("edge");
		for(int j = 0; j < hostAlgorithm->CHROMOSOME_SIZE; j++){
			double* currentCostHost = &(hostAlgorithm->adjacencyMatrix[i*hostAlgorithm->CHROMOSOME_SIZE+j]);

			if(i == j){
				*currentCostHost = 0;
			}else{
				cost = edge->first_attribute("cost");
				*currentCostHost = *(cost->value());
			}
		}
	}
}

void runGeneticAlgorithm(){
	createRandomPermutation
		<<<
			hostAlgorithm->GRID_SIZE,
			hostAlgorithm->BLOCK_SIZE,
			(hostAlgorithm->ISLAND_POPULATION_SIZE*hostAlgorithm->CHROMOSOME_SIZE*sizeof(int))
		>>>
		(deviceAlgorithm);


	createRandomSeeds<<<
			hostAlgorithm->GRID_SIZE,
			hostAlgorithm->BLOCK_SIZE
		>>>(deviceAlgorithm, time(NULL));
	check(cudaDeviceSynchronize());


	for(int i = 0; i < hostAlgorithm->GENERATIONS; i++){
		runOneGeneration
			<<<
			hostAlgorithm->GRID_SIZE,
			hostAlgorithm->BLOCK_SIZE,
			(hostAlgorithm->ISLAND_POPULATION_SIZE*hostAlgorithm->CHROMOSOME_SIZE*sizeof(int) + hostAlgorithm->ISLAND_POPULATION_SIZE*sizeof(double))
			>>>
			(deviceAlgorithm);
		check(cudaDeviceSynchronize());
	}
}

int main(int argc, char ** argv){
	try{
		char* filename = argv[1];
		int blockSize = atoi(argv[2]);
		int gridSize = atoi(argv[3]);
		int generations = atoi(argv[4]);


		rapidxml::xml_document<> doc;
		rapidxml::file<> file(filename);
		doc.parse<0>(file.data());
		rapidxml::xml_node<>* graph = doc.first_node("travellingSalesmanProblemInstance")->first_node("graph");


		hostAlgorithm = new geneticAlgorithm(blockSize, gridSize, generations, rapidxml::count_children(graph));
		geneticAlgorithm * hostDevice = new geneticAlgorithm(blockSize, gridSize, generations, rapidxml::count_children(graph));
		check(cudaMalloc((void**) &deviceAlgorithm, sizeof(geneticAlgorithm)));

		(hostAlgorithm->seeds) = new int[hostAlgorithm->POPULATION_SIZE];
		cudaMalloc((void**) &(hostDevice->seeds), hostAlgorithm->POPULATION_SIZE*sizeof(long));

		(hostAlgorithm->source) = new int[hostAlgorithm->CHROMOSOME_SIZE];
		cudaMalloc((void**) &(hostDevice->source), hostAlgorithm->CHROMOSOME_SIZE*sizeof(int));

		(hostAlgorithm->adjacencyMatrix) = new double[hostAlgorithm->CHROMOSOME_SIZE*hostAlgorithm->CHROMOSOME_SIZE];
		cudaMalloc((void**) &(hostDevice->adjacencyMatrix), hostAlgorithm->CHROMOSOME_SIZE*hostAlgorithm->CHROMOSOME_SIZE*sizeof(double));

		(hostAlgorithm->populationChromosome) = new int[hostAlgorithm->POPULATION_SIZE*hostAlgorithm->CHROMOSOME_SIZE];
		cudaMalloc((void**) &(hostDevice->populationChromosome), sizeof(int)*hostAlgorithm->POPULATION_SIZE*hostAlgorithm->CHROMOSOME_SIZE);

		(hostAlgorithm->populationDistance) = new double[hostAlgorithm->POPULATION_SIZE];
		cudaMalloc((void**) &(hostDevice->populationDistance), sizeof(double)*hostAlgorithm->POPULATION_SIZE);

		readDataFromXMLInstance(graph);
		check(cudaMemcpy(hostDevice->adjacencyMatrix, hostAlgorithm->adjacencyMatrix, sizeof(int)*hostAlgorithm->CHROMOSOME_SIZE, cudaMemcpyHostToDevice));


		for(int i = 0; i < hostAlgorithm->CHROMOSOME_SIZE; i++){
			hostAlgorithm->source[i] = i;
		}

		check(cudaMemcpy(hostDevice->source, hostAlgorithm->source, sizeof(int)*hostAlgorithm->CHROMOSOME_SIZE, cudaMemcpyHostToDevice));
		check(cudaMemcpy(deviceAlgorithm, hostDevice, sizeof(geneticAlgorithm), cudaMemcpyHostToDevice));


		runGeneticAlgorithm();

		check(cudaMemcpy(hostDevice->populationDistance, hostAlgorithm->populationDistance, sizeof(int)*hostAlgorithm->CHROMOSOME_SIZE, cudaMemcpyDeviceToHost));
		check(cudaMemcpy(hostDevice->populationChromosome, hostAlgorithm->populationChromosome, sizeof(int)*hostAlgorithm->POPULATION_SIZE*hostAlgorithm->CHROMOSOME_SIZE, cudaMemcpyDeviceToHost));

		for (int i = 0; i < hostAlgorithm->POPULATION_SIZE; i++){
			std::cout << '[' << chromosomeCheck(&(hostAlgorithm->populationChromosome[i*hostAlgorithm->CHROMOSOME_SIZE])) << ']' << " ";
			for(int j = 0; j < hostAlgorithm->CHROMOSOME_SIZE; j++){
				std::cout << hostAlgorithm->populationChromosome[i*hostAlgorithm->CHROMOSOME_SIZE+j] << " ";
			}
			std::cout << hostAlgorithm->populationDistance[i] << std::endl;
		}


		free(hostAlgorithm->seeds);
		free(hostAlgorithm->source);
		free(hostAlgorithm->adjacencyMatrix);
		free(hostAlgorithm->populationChromosome);
		free(hostAlgorithm->populationDistance);
		free(hostAlgorithm);
		cudaFree(hostDevice->seeds);
		cudaFree(hostDevice->source);
		cudaFree(hostDevice->adjacencyMatrix);
		cudaFree(hostDevice->populationChromosome);
		cudaFree(hostDevice->populationDistance);
		cudaFree(hostDevice);
		cudaFree(deviceAlgorithm);

	}
	catch(cudaError * e){
		std::cout << "Oh crap: " << cudaGetErrorString(*e) << std::endl;
	}
}



