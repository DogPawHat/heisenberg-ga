#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include "global_structs.h"
#include "berlin52.h"

__global__ void createRandomPermutation(deviceFields fields, long seed);
__global__ void runGeneticAlgorithm(deviceFields fields);
__global__ void createRandomSeeds(deviceFields fields, long seed);

int main(){
	deviceFields device;
	hostFields host;

	cudaMalloc((void**) &device.population, TOTAL_POPULATION_MEMORY_SIZE*sizeof(short));
	cudaMalloc((void**) &device.seeds, POPULATION_SIZE*sizeof(int));
	cudaMalloc((void**) &device.TSPGraph, 2*CHROMOSOME_SIZE*sizeof(float));
	cudaMalloc((void**) &device.source, CHROMOSOME_SIZE*sizeof(short));

	cudaMemcpy(device.TSPGraph, berlin52, 2*CHROMOSOME_SIZE*sizeof(float), cudaMemcpyHostToDevice);
	
	thrust::device_ptr<short> sourceThrust(device.source);
	thrust::sequence(sourceThrust, sourceThrust+CHROMOSOME_SIZE);

	createRandomSeeds<<<GRID_SIZE, BLOCK_SIZE>>>(device, time(NULL));
	createRandomPermutation<<<GRID_SIZE, BLOCK_SIZE>>>(device, time(NULL));
	cudaDeviceSynchronize();

	runGeneticAlgorithm<<<GRID_SIZE, BLOCK_SIZE>>>(device);
	cudaDeviceSynchronize();

	cudaMemcpy(host.population, device.population, TOTAL_POPULATION_MEMORY_SIZE*sizeof(short),cudaMemcpyDeviceToHost);

	for (short i = 0; i < POPULATION_SIZE; i++){
		for(short j = 0; j < CHROMOSOME_SIZE; j++){
			std::cout << host.population[j+i*CHROMOSOME_SIZE] << " ";
		}
		std::cout << std::endl;
	}

	cudaFree(device.population);
	cudaFree(device.population);

	std::cin.get();
}