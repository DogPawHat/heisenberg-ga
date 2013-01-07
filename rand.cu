#include <cuda.h>
#include <curand_kernel.h>


//Create an random int array repesenting a solution to a TSP. For inisziation.
__device__ int* rand(int* start, curandState* devStates){
	int tx = threadIdx.x;
	for(unsigned int i = 0; i < 52; i++){
		curand_init(456723, tx, 0, &devStates[i]);
	}
	int result[52];
	int rand;
	for(unsigned int i = 0; i < 52; i++){
		int rand = floor(curand_uniform(&devStates[i]) * (52-i));
		result[i] = start[rand];
		start[i] = start[52 - i];
	}

	return result;
}

__global__ void testRand(int* test, curandState* devStates){
	int start[52];
	for(int i = 0; i < 52; i++){
		start[i] = i;
	}
	test = start;
}