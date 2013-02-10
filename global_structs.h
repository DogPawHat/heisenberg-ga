#ifndef GLOBAL_STRUCTS
#define GLOBAL_STRUCTS

typedef struct{
	int * population;
	int * source;
	int * TSPGraph;
} deviceFields;

typedef struct{
	int * population;
} hostFields;

typedef struct{
	int populationSize, islandPopulationSize, chromosomeSize;
} fieldSizes;

#endif