#ifndef GLOBAL_STRUCTS
#define GLOBAL_STRUCTS

const int BLOCK_SIZE = 64;
const int GRID_SIZE = 10;
const int GENERATIONS = 10;
const int CHROMOSOME_SIZE = setChromosomeSize();
const int POPULATION_SIZE = (BLOCK_SIZE*GRID_SIZE);
const int ISLAND_POPULATION_SIZE = BLOCK_SIZE;
const int CROSSOVER_CHANCE = 60;
const int MUTATION_CHANCE = 10;


const int setChromosomeSize(int set){
	return set;
}

typedef struct{
	int chromosome[CHROMOSOME_SIZE];
	long distance;
	__host__ __device__ int distanceCalculation(int TSPGraph[]){
		distance = distanceBetweenTwoCities(chromosome[CHROMOSOME_SIZE-1], chromosome[0], TSPGraph);
		for(unsigned int i = 1; i < CHROMOSOME_SIZE; i++){
			unsigned int j  = i - 1;
			distance += distanceBetweenTwoCities(chromosome[i], chromosome[j], TSPGraph);
		}
		return distance;
	}
	__host__ __device__ long distanceBetweenTwoCities(int i, int j, int TSPGraph[]){
		double xi = TSPGraph[2*i];
		double xj = TSPGraph[2*j];
		double yi = TSPGraph[2*i+1];
		double yj = TSPGraph[2*j+1];
		double xd = abs(xi - xj);
		double yd = abs(yi - yj);
		return(lrint(sqrt(xd*xd + yd*yd)));
	}
} metaChromosome;

typedef struct{
	metaChromosome population[POPULATION_SIZE];
	int source[CHROMOSOME_SIZE];
	int seeds[POPULATION_SIZE];
	int TSPGraph[2*CHROMOSOME_SIZE];
} deviceFields;

typedef struct{
	double * adjacencyMatrix;
} TSPGraph;

#endif
