#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include "global_structs.h"
#include "berlin52.h"

__global__ void createRandomPermutation(deviceFields fields, long seed);
__global__ void runGeneticAlgorithm(deviceFields fields);

int main(){
	deviceFields device;
	hostFields host;

	host.population = (short*) malloc(TOTAL_POPULATION_MEMORY_SIZE*sizeof(short));
	cudaMalloc((void**) &device.source, CHROMOSOME_SIZE*sizeof(short));
	cudaMalloc((void**) &device.population,  TOTAL_POPULATION_MEMORY_SIZE*sizeof(short));
	cudaMalloc((void**) &device.TSPGraph, CHROMOSOME_SIZE*2*sizeof(float));
	cudaMemcpy(device.TSPGraph, berlin52, CHROMOSOME_SIZE*2*sizeof(float), cudaMemcpyHostToDevice);
	thrust::device_ptr<short> sourceThrust = thrust::device_pointer_cast(device.source);
	thrust::sequence(sourceThrust, sourceThrust + CHROMOSOME_SIZE);

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
	cudaFree(device.TSPGraph);
	cudaFree(device.source);
	free(host.population);
	std::cin.get();
}