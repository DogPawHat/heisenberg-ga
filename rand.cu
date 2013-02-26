#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include "global_structs.h"

using thrust::random::minstd_rand0;
using thrust::random::uniform_int_distribution;

//Create an random int array repesenting a solution to a TSP. For inisziation.
__global__ void createRandomPermutation(deviceFields fields, long seed){
	short tempResult[CHROMOSOME_SIZE];
	short temp;
	short rand;
	short start = (threadIdx.x + blockIdx.x*blockDim.x)*CHROMOSOME_SIZE;

	minstd_rand0 rng(seed*(threadIdx.x + blockIdx.x*blockDim.x)-341256);

	for(short i = 0; i < CHROMOSOME_SIZE; i++){
		tempResult[i] = fields.source[i];
	}

	for(short i = CHROMOSOME_SIZE-1; i > 0; i--){
		uniform_int_distribution<short> dist(0,i+1);
		rand = dist(rng);
		temp = tempResult[rand];
		tempResult[rand] = tempResult[i];
		tempResult[i] = temp;
	}

	for(short i = 0; i < CHROMOSOME_SIZE; i++){
		fields.population[start+i] = tempResult[i];
	}
}

__global__ void createRandomSeeds(deviceFields fields, long seed){
	minstd_rand0 rng(seed*(threadIdx.x + blockIdx.x*blockDim.x)-34156);

	uniform_int_distribution<int> dist(0,RAND_MAX);
	fields.seeds[threadIdx.x + blockDim.x*blockIdx.x]=dist(rng);
}