#ifndef GLOBAL_STRUCTS
#define GLOBAL_STRUCTS

#define BLOCK_SIZE 64
#define GRID_SIZE 10
#define GENERATIONS 1
#define CHROMOSOME_SIZE 52
#define POPULATION_SIZE (BLOCK_SIZE*GRID_SIZE)
#define ISLAND_POPULATION_SIZE BLOCK_SIZE
#define CROSSOVER_CHANCE 60
#define MUTATION_CHANCE 10

typedef struct{
	int chromosome[CHROMOSOME_SIZE];
	int distance;
	__host__ __device__ int distanceCalculation(int TSPGraph[]){
		distance = distanceBetweenTwoCities(chromosome[CHROMOSOME_SIZE-1], chromosome[0], TSPGraph);
		for(unsigned int i = 1; i < CHROMOSOME_SIZE; i++){
			unsigned int j  = i - 1;
			distance += distanceBetweenTwoCities(chromosome[i], chromosome[j], TSPGraph);
		}
		return distance;
	}

	__host__ __device__ int distanceBetweenTwoCities(short i, short j, int TSPGraph[]){
		int xi = TSPGraph[2*i];
		int xj = TSPGraph[2*j];
		int yi = TSPGraph[2*i+1];
		int yj = TSPGraph[2*j+1];
		int xd = abs(xi - xj);
		int yd = abs(yi - yj);
		return(rintf(sqrtf(xd*xd + yd*yd)));
	}
} metaChromosome;

typedef struct{
	metaChromosome population[POPULATION_SIZE];
	int source[CHROMOSOME_SIZE];
	int seeds[POPULATION_SIZE];
	int TSPGraph[2*CHROMOSOME_SIZE];
} deviceFields;

#endif
