#include "cuda.h"
#include <curand_kernel.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include "rand.h"

#define BLOCK_SIZE 32
#define GRID_SIZE 10

int main(){
	const int testSize = 52;
	int* source;
	int* devicePopulation;
	int hostPopulation[testSize*(BLOCK_SIZE*GRID_SIZE)];

	cudaMalloc((void**) &source, testSize*sizeof(int));
	cudaMalloc((void**) &devicePopulation, testSize*(BLOCK_SIZE*GRID_SIZE)*sizeof(int));
	thrust::device_ptr<int> sourceThrust = thrust::device_pointer_cast(source);
	thrust::sequence(sourceThrust, sourceThrust + testSize);

	createRandomPermutation<<<GRID_SIZE, BLOCK_SIZE>>>(source, deviceResult, testSize, time(NULL));
	cudaMemcpy(hostPopulation, devicePopulation, testSize*(BLOCK_SIZE*GRID_SIZE)*sizeof(int),cudaMemcpyDeviceToHost);
	for (int i = 0; i < BLOCK_SIZE*GRID_SIZE; i++){
		for(int j = 0; j < testSize; j++){
			std::cout << hostPopulation[j+i*testSize] << " ";
		}
		std::cout << std::endl;
	}

	cudaFree(devicePopulation);
	cudaFree(source);
	std::cin.get();
}