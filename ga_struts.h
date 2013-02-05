#ifndef GA_STRUTS_H
#define GA_STRUTS_H

#include<cuda.h>

typedef struct{
	const int populationSize;
	const int chromosomeSize;
} fieldSizes;


typedef struct {
	int* population;
	int* TSPGraph;
	int* source;
} deviceFields;

typedef struct{
	int* population;
} hostFields;

#endif