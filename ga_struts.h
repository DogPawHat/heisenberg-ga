#ifndef GA_STRUTS_H
#define GA_STRUTS_H

struct fieldSizes{
	const int populationSize;
	const int chromosomeSize;
}


struct deviceFields{
	int* population;
	int* TSPGraph;
	int* source;
}

struct hostFields{
	int* population;
}

#endif