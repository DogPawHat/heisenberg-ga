#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>

using thrust::random::minstd_rand0;
using thrust::random::uniform_int_distribution;

//Create an random int array repesenting a solution to a TSP. For inisziation.
__global__ void createRandomPermutation(int* source, int* result, int size, long seed){
	int rand;
	int start = (threadIdx.x + blockIdx.x*blockDim.x)*size;
	int* tempSource = (int*) malloc(sizeof(int)*size);
	memcpy(tempSource, source, sizeof(int)*size);
	minstd_rand0 rng(seed+(threadIdx.x + blockIdx.x*blockDim.x));
	for(int i = 0; i < size; i++){
		uniform_int_distribution<int> dist(0,(size-i));
		rand = dist(rng);
		result[start+i] = tempSource[rand];
		tempSource[rand] = tempSource[(size-i)-1];
	}
	free(tempSource);
}