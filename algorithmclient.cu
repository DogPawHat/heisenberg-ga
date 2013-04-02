#include "rapidxml.hpp"
#include "rapidxml_utils.hpp"
#include "algorithm.cuh"


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
	createRandomPermutation<<<GRID_SIZE, BLOCK_SIZE,(deviceAlgorithm->ISLAND_POPULATION_SIZE*deviceAlgorithm->CHROMOSOME_SIZE*sizeof(int))>>>(*deviceAlgorithm);
	createRandomSeeds<<<GRID_SIZE, BLOCK_SIZE>>>(*deviceAlgorithm, time(NULL));
	check(cudaDeviceSynchronize());


	for(int i = 0; i < deviceAlgorithm->GENERATIONS; i++){
		runOneGeneration
			<<<
			GRID_SIZE, BLOCK_SIZE,
			((deviceAlgorithm->ISLAND_POPULATION_SIZE)*(deviceAlgorithm->CHROMOSOME_SIZE)*sizeof(int))
			>>>
			(*deviceAlgorithm);
		check(cudaDeviceSynchronize());
	}

}

int main(int argc, char ** argv){
	try{
		char* filename = argv[1];
		int generations = atoi(argv[4]);


		rapidxml::xml_document<> doc;
		rapidxml::file<> file(filename);
		doc.parse<0>(file.data());
		rapidxml::xml_node<>* graph = doc.first_node("travellingSalesmanProblemInstance")->first_node("graph");


		geneticAlgorithm * hostAlgorithm = new geneticAlgorithm(generations, rapidxml::count_children(graph));
		geneticAlgorithm * deviceAlgorithm = new geneticAlgorithm(generations, rapidxml::count_children(graph));

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


		for(int i = 0; i < hostAlgorithm->CHROMOSOME_SIZE; i++){
			hostAlgorithm->source[i] = i;
		}

		check(cudaMemcpy(deviceAlgorithm->source, hostAlgorithm->source, sizeof(int)*hostAlgorithm->CHROMOSOME_SIZE, cudaMemcpyHostToDevice));


		runGeneticAlgorithm(deviceAlgorithm);

		check(cudaMemcpy(hostAlgorithm->populationDistance ,deviceAlgorithm->populationDistance, sizeof(double)*hostAlgorithm->POPULATION_SIZE, cudaMemcpyDeviceToHost));
		check(cudaMemcpy(hostAlgorithm->populationChromosome ,deviceAlgorithm->populationChromosome, sizeof(int)*hostAlgorithm->POPULATION_SIZE*hostAlgorithm->CHROMOSOME_SIZE, cudaMemcpyDeviceToHost));

		for (int i = 0; i < hostAlgorithm->POPULATION_SIZE; i++){
			std::cout << '[' << chromosomeCheck(&(hostAlgorithm->populationChromosome[i*hostAlgorithm->CHROMOSOME_SIZE]), hostAlgorithm) << ']' << " ";
			for(int j = 0; j < hostAlgorithm->CHROMOSOME_SIZE; j++){
				std::cout << hostAlgorithm->populationChromosome[i*hostAlgorithm->CHROMOSOME_SIZE+j] << " ";
			}
			std::cout << hostAlgorithm->populationDistance[i] << std::endl;
		}

		bool allGood = true;
		for(int i = 0; i < hostAlgorithm->POPULATION_SIZE; i++){
			if(chromosomeCheck(&(hostAlgorithm->populationChromosome[i*hostAlgorithm->CHROMOSOME_SIZE]), hostAlgorithm) == 1){
				allGood = false;
				break;
			}
		}

		if(allGood){
			std::cout << "All good Captian";
		}else{
			std::cout << "Well Hitlerfuck";
		}

		free(hostAlgorithm->seeds);
		free(hostAlgorithm->source);
		free(hostAlgorithm->adjacencyMatrix);
		free(hostAlgorithm->populationChromosome);
		free(hostAlgorithm->populationDistance);
		free(hostAlgorithm);
		cudaFree(deviceAlgorithm->seeds);
		cudaFree(deviceAlgorithm->source);
		cudaFree(deviceAlgorithm->adjacencyMatrix);
		cudaFree(deviceAlgorithm->populationChromosome);
		cudaFree(deviceAlgorithm->populationDistance);
		cudaFree(deviceAlgorithm);

	}
	catch(cudaError * e){
		std::cout << "Oh crap: " << cudaGetErrorString(*e) << std::endl;
	}
}



