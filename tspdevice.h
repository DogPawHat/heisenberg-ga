#ifndef TSPDEVICE_H
#define TSPDEVICE_H

#include "cuda.h"


struct TSPDevice{
	double** list;
	int size;
};

__device__ double GetDistance(double*, double*);

#endif