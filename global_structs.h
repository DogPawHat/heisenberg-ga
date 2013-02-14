#ifndef GLOBAL_STRUCTS
#define GLOBAL_STRUCTS

#define BLOCK_SIZE 2
#define GRID_SIZE 2
#define CHROMOSOME_SIZE 52
#define POPULATION_SIZE (BLOCK_SIZE*GRID_SIZE)
#define ISLAND_POPULATION_SIZE BLOCK_SIZE
#define TOTAL_POPULATION_MEMORY_SIZE (POPULATION_SIZE*CHROMOSOME_SIZE)
#define TOTAL_ISLAND_POPULATION_MEMORY_SIZE (ISLAND_POPULATION_SIZE*CHROMOSOME_SIZE) 

typedef struct{
	short * population; //shorts are only 2 bytes long
	short * source;
	short * TSPGraph;
} deviceFields;

typedef struct{
	short * population;
} hostFields;

#endif