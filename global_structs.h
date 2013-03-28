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
	short chromosome[CHROMOSOME_SIZE];
	float distance;
	__host__ __device__ float distanceCalculation(float TSPGraph[]){
		distance = distanceBetweenTwoCities(chromosome[CHROMOSOME_SIZE-1], chromosome[0], TSPGraph);
		for(short i = 1; i < CHROMOSOME_SIZE; i++){
			short j  = i - 1;
			distance += distanceBetweenTwoCities(chromosome[i], chromosome[j], TSPGraph);
		}
		return distance;
	}

	__host__ __device__ float distanceBetweenTwoCities(short i, short j, float TSPGraph[]){
		float xi = TSPGraph[2*i];
		float xj = TSPGraph[2*j];
		float yi = TSPGraph[2*i+1];
		float yj = TSPGraph[2*j+1];
		float xd = fmaxf(xi, xj) - fminf(xi, xj);
		float yd = fmaxf(yi, yj) - fminf(yi, yj);
		return(sqrtf(xd*xd + yd*yd));
	}


	float fitness;
} metaChromosome;

typedef struct{
	metaChromosome population[POPULATION_SIZE];
	short source[CHROMOSOME_SIZE];
	int seeds[POPULATION_SIZE];
	float TSPGraph[2*CHROMOSOME_SIZE];
} deviceFields;

#endif
