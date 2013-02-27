#ifndef GLOBAL_STRUCTS
#define GLOBAL_STRUCTS

#define BLOCK_SIZE 32
#define GRID_SIZE 10
#define CHROMOSOME_SIZE 52
#define POPULATION_SIZE (BLOCK_SIZE*GRID_SIZE)
#define ISLAND_POPULATION_SIZE BLOCK_SIZE
#define MATING_POOL_SIZE (ISLAND_POPULATION_SIZE/8)
#define TOTAL_POPULATION_MEMORY_SIZE (POPULATION_SIZE*CHROMOSOME_SIZE)
#define TOTAL_ISLAND_POPULATION_MEMORY_SIZE (ISLAND_POPULATION_SIZE*CHROMOSOME_SIZE)
#define TOTAL_MATING_POOL_MEMORY_SIZE (MATING_POOL_SIZE*CHROMOSOME_SIZE)

typedef struct{
	short * population; //shorts are only 2 bytes long
	short * source;
	short * TSPGraph;
	int * seeds;
} deviceFields;

typedef struct{
	short * population;
	int * seeds;
} hostFields;

#endif
