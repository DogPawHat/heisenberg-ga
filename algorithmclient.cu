#include "rapidxml.hpp"
#include "rapidxml_utils.hpp"
#include "algorithm.cuh"


geneticAlgorithm * hostAlgorithm;
geneticAlgorithm * deviceAlgorithm;


int chromosomeCheck(int chromosome[]){
	int k;
	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		k = 0;
		for(int j = 0; j < CHROMOSOME_SIZE; j++){
			if(chromosome[j] == i){
				k++;
			}else if(chromosome[j] > CHROMOSOME_SIZE || chromosome[j] < 0){
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
			double* currentCostDevice = &(deviceAlgorithm->adjacencyMatrix[i*hostAlgorithm->CHROMOSOME_SIZE+j]);

			if(i == j){
				*currentCostHost = 0;
				*currentCostDevice = 0;
			}else{
				cost = edge->first_attribute("cost");
				*currentCostHost = *(cost->value());
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
	createRandomSeeds(deviceAlgorithm, time(NULL))<<<
			hostAlgorithm->GRID_SIZE,
			hostAlgorithm->BLOCK_SIZE
		>>>;

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
	char* filename = argv[1];
	int gridSize = argv[2];
	int blockSize = argv[3];
	int generations = argv[4];


	rapidxml::xml_document<> doc;
	rapidxml::file<> file(filename);
	doc.parse<0>(file.data());
	rapidxml::xml_node<>* graph = doc.first_node("graph");


	hostAlgorithm = new geneticAlgorithm(gridSize, blockSize, generations, rapidxml::count_children(graph));
	cudaMalloc((void**) &deviceAlgorithm, sizeof(geneticAlgorithm));
	cudaMemcpy(deviceAlgorithm, hostAlgorithm, sizeof(geneticAlgorithm), cudaMemcpyHostToDevice);

	hostAlgorithm->source = new int[hostAlgorithm->CHROMOSOME_SIZE];
	cudaMalloc((void**) &(deviceAlgorithm->source), hostAlgorithm->CHROMOSOME_SIZE*sizeof(int));

	hostAlgorithm->seeds = new long[hostAlgorithm->CHROMOSOME_SIZE];
	cudaMalloc((void**) &(deviceAlgorithm->seeds), hostAlgorithm->CHROMOSOME_SIZE*sizeof(long));
	
	hostAlgorithm->adjacencyMatrix = new double[hostAlgorithm->CHROMOSOME_SIZE*hostAlgorithm->CHROMOSOME_SIZE];
	cudaMalloc((void**) &(deviceAlgorithm->adjacencyMatrix), hostAlgorithm->CHROMOSOME_SIZE*hostAlgorithm->CHROMOSOME_SIZE*sizeof(double));
	
	hostAlgorithm->populationChromosome = new int[hostAlgorithm->POPULATION_SIZE*hostAlgorithm->CHROMOSOME_SIZE];
	cudaMalloc((void**) &(deviceAlgorithm->populationChromosome), sizeof(int)*hostAlgorithm->POPULATION_SIZE*hostAlgorithm->CHROMOSOME_SIZE);
	
	hostAlgorithm->populationDistance = new double[hostAlgorithm->POPULATION_SIZE];
	cudaMalloc((void**) &(deviceAlgorithm->populationChromosome), sizeof(double)*hostAlgorithm->POPULATION_SIZE);

	readDataFromXMLInstance(graph);

	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		hostAlgorithm->source[i] = i;
		hostAlgorithm->TSPGraph[i*2] = berlin52[i*2];
		hostAlgorithm->TSPGraph[i*2 + 1] = berlin52[i*2 + 1];
	}

	cudaMemcpy((deviceAlgorithm->source, hostAlgorithm->source, sizeof(int), cudaMemcpyHostToDevice);


}



