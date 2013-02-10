#ifndef GLOBAL_STRUCTS
#define GLOBAL_STRUCTS

struct deviceFields{
	int * population;
	int * source;
	int * TSPGraph;
};

struct hostFields{
	int * population;
};

struct fieldSizes
{
	int populationSize, islandPopulationSize, chromosomeSize;
};

#endif