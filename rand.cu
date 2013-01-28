#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>

using thrust::random::minstd_rand0;
using thrust::random::uniform_int_distribution;

//Create an random int array repesenting a solution to a TSP. For inisziation.
__global__ void createRandomPermutation(int* source, int* result, const int chromosomeSize, long seed){
	int rand;
	int start = (threadIdx.x + blockIdx.x*blockDim.x)*chromosomeSize;
	int* tempSource = (int*) malloc(sizeof(int)*chromosomeSize);
	memcpy(tempSource, source, sizeof(int)*chromosomeSize);
	minstd_rand0 rng(seed+(threadIdx.x + blockIdx.x*blockDim.x));
	for(int i = 0; i < chromosomeSize; i++){
		uniform_int_distribution<int> dist(0,(chromosomeSize-i));
		rand = dist(rng);
		result[start+i] = tempSource[rand];
		tempSource[rand] = tempSource[(chromosomeSize-i)-1];
	}
	free(tempSource);
}