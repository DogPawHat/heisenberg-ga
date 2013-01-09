#include <cuda.h>
#include <curand_kernel.h>


//Create an random int array repesenting a solution to a TSP. For inisziation.
__device__ void doRand(int start[], int size, curandState devStates[]){
	int tx = threadIdx.x;
	for(unsigned int i = 0; i < size; i++){
		curand_init(456723, tx, 0, &devStates[i]);
	}
	int * result = (int*) malloc(size*sizeof(int));
	int rand;
	for(unsigned int i = 0; i < 52; i++){
		rand = floor(curand_uniform(&devStates[i]) * (52-i));
		result[i] = start[rand];
		start[rand] = start[size - i];
	}
	memcpy(start, result, size*sizeof(int));
}

__global__ void testRand(float test[], const int size){
	curandState *devStates = (curandState*) malloc(size*sizeof(curandState));
	int tx = threadIdx.x;
	for(unsigned int i = 0; i < size; i++){
		curand_init(456723, tx, 0, &devStates[i]);
	}
	for(unsigned int i = 0; i < 52; i++){
		test[i] = curand_uniform(&devStates[i]);
	}
	free(devStates);
}