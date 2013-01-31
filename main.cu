#include "cuda.h"
#include <curand_kernel.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include "berlin52.h"
#include "rand.h"

#define BLOCK_SIZE 32
#define GRID_SIZE 10
#define POPULATION_MULTIPLIER 5

int main(){
	const int gridSize = 10;
	const int populationSize = BLOCK_SIZE*GRID_SIZE;
	const int chromosomeSize = 52;
	int* source;
	int* devicePopulation;
	int* deviceTSPRoute;
	int* hostPopulation;

	cudaMallocHost((void**) &hostPopulation, populationSize*sizeof(int));
	cudaMalloc((void**) &source, chromosomeSize*sizeof(int));
	cudaMalloc((void**) &devicePopulation, populationSize*sizeof(int));
	cudaMalloc((void**) &deviceTSPRoute, chromosomeSize*2*sizeof(float));
	cudaMemcpy(deviceTSPRoute, berlin52, chromosomeSize*2*sizeof(float), cudaMemcpyHostToDevice);
	thrust::device_ptr<int> sourceThrust = thrust::device_pointer_cast(source);
	thrust::sequence(sourceThrust, sourceThrust + chromosomeSize);

	createRandomPermutation<<<gridSize, BLOCK_SIZE>>>(source, devicePopulation, chromosomeSize, time(NULL));
	cudaDeviceSynchronize();

	cudaMemcpy(hostPopulation, devicePopulation, populationSize*sizeof(int),cudaMemcpyDeviceToHost);
	for (int i = 0; i < populationSize; i++){
		for(int j = 0; j < chromosomeSize; j++){
			std::cout << hostPopulation[j+i*chromosomeSize] << " ";
		}
		std::cout << std::endl;
	}

	cudaFree(devicePopulation);
	cudaFree(deviceTSPRoute);
	cudaFree(source);
	cudaHostFree(hostPopulation);
	std::cin.get();
}