#ifndef GLOBAL_STRUCTS
#define GLOBAL_STRUCTS

#define BLOCK_SIZE 64
#define GRID_SIZE 10
#define GENERATIONS 1
#define CHROMOSOME_SIZE 52
#define POPULATION_SIZE (BLOCK_SIZE*GRID_SIZE)
#define ISLAND_POPULATION_SIZE BLOCK_SIZE

typedef struct{
	short chromosome[CHROMOSOME_SIZE];
	float distance;
	__host__ __device__ float distanceCalculation(float TSPGraph[]){
		distance = 0;
		for(short i = 1; i < CHROMOSOME_SIZE; i++){
			short j  = i - 1;
			float xi = TSPGraph[2*chromosome[i]];
			float xj = TSPGraph[2*chromosome[j]];
			float yi = TSPGraph[2*chromosome[i]+1];
			float yj = TSPGraph[2*chromosome[j]+1];
			float xd = fmaxf(xi, xj) - fminf(xi, xj);
			float yd = fmaxf(yi, yj) - fminf(yi, yj);
			distance += sqrtf(xd*xd + yd*yd);
		}
		return distance;
	}
	float fitness;
} metaChromosome;

typedef struct{
	metaChromosome * population; //shorts are only 2 bytes long
	short * source;
	float * TSPGraph;
	int * seeds;
} deviceFields;

/*typedef struct{
	short * population; //shorts are only 2 bytes long
	short * source;
	float * TSPGraph;
	int * seeds;
} deviceFields;
*/

typedef struct{
	metaChromosome population[POPULATION_SIZE];
	short source[CHROMOSOME_SIZE];
	int seeds[POPULATION_SIZE];
} hostFields;

#endif
