#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include "ga_struts.h"

using thrust::random::minstd_rand0;
using thrust::random::uniform_int_distribution;

//Create an random int array repesenting a solution to a TSP. For inisziation.
__global__ void createRandomPermutation(fieldSizes sizes, deviceFields fields, long seed){
	int rand;
	int start = (threadIdx.x + blockIdx.x*blockDim.x)*sizes.chromosomeSize;
	int* tempSource = (int*) malloc(sizeof(int)*sizes.chromosomeSize);
	memcpy(tempSource, fields.source, sizeof(int)*sizes.chromosomeSize);
	minstd_rand0 rng(seed+(threadIdx.x + blockIdx.x*blockDim.x));
	for(int i = 0; i < sizes.chromosomeSize; i++){
		uniform_int_distribution<int> dist(0,(sizes.chromosomeSize-i));
		rand = dist(rng);
		fields.population[start+i] = tempSource[rand];
		tempSource[rand] = tempSource[(sizes.chromosomeSize-i)-1];
	}
	free(tempSource);
}