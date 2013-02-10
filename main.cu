#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include "global_structs.h"
#include "berlin52.h"

#define BLOCK_SIZE 256
#define GRID_SIZE 10


__global__ void createRandomPermutation(fieldSizes sizes, deviceFields fields, long seed);
__global__ void runGeneticAlgorithm(deviceFields fields, fieldSizes sizes);

int main(){
	deviceFields device;
	hostFields host;
	fieldSizes sizes;
	sizes.populationSize = BLOCK_SIZE*GRID_SIZE;
	sizes.islandPopulationSize = BLOCK_SIZE;
	sizes.chromosomeSize = 52;

	cudaMallocHost((void**) &host.population, sizes.populationSize*sizeof(int));
	cudaMalloc((void**) &device.source, sizes.chromosomeSize*sizeof(int));
	cudaMalloc((void**) &device.population, sizes.populationSize*sizeof(int));
	cudaMalloc((void**) &device.TSPGraph, sizes.chromosomeSize*2*sizeof(float));
	cudaMemcpy(device.TSPGraph, berlin52, sizes.chromosomeSize*2*sizeof(float), cudaMemcpyHostToDevice);
	thrust::device_ptr<int> sourceThrust = thrust::device_pointer_cast(device.source);
	thrust::sequence(sourceThrust, sourceThrust + sizes.chromosomeSize);

	createRandomPermutation<<<GRID_SIZE, BLOCK_SIZE>>>(sizes, device, time(NULL));
	cudaDeviceSynchronize();

	cudaMemcpy(host.population, device.population, sizes.populationSize*sizeof(int),cudaMemcpyDeviceToHost);

	for (int i = 0; i < sizes.populationSize; i++){
		for(int j = 0; j < sizes.chromosomeSize; j++){
			std::cout << host.population[j+i*sizes.chromosomeSize] << " ";
		}
		std::cout << std::endl;
	}

	cudaFree(device.population);
	cudaFree(device.TSPGraph);
	cudaFree(device.source);
	cudaFreeHost(host.population);
	std::cin.get();
}