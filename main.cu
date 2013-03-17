#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include "global_structs.h"
#include "berlin52.h"

__global__ void createRandomPermutation(deviceFields fields);
__global__ void runGeneticAlgorithm(deviceFields fields);
__global__ void createRandomSeeds(deviceFields fields, long seed);

cudaError check(cudaError call){
	if(call != cudaSuccess){
		throw call;
	}
}

int main(){
	try{
		deviceFields device;
		hostFields host;

		cudaMalloc((void**) &device.population, POPULATION_SIZE*sizeof(metaChromosome));
		cudaMalloc((void**) &device.seeds, POPULATION_SIZE*sizeof(int));
		cudaMalloc((void**) &device.TSPGraph, 2*CHROMOSOME_SIZE*sizeof(float));
		cudaMalloc((void**) &device.source, CHROMOSOME_SIZE*sizeof(short));

		cudaMemcpy(device.TSPGraph, berlin52, 2*CHROMOSOME_SIZE*sizeof(float), cudaMemcpyHostToDevice);

		check(cudaThreadSynchronize());
	
		thrust::device_ptr<short> sourceThrust(device.source);
		thrust::sequence(sourceThrust, sourceThrust+CHROMOSOME_SIZE);

		createRandomSeeds<<<GRID_SIZE, BLOCK_SIZE>>>(device, time(NULL));
		createRandomPermutation<<<GRID_SIZE, BLOCK_SIZE>>>(device);
		check(cudaThreadSynchronize());

		runGeneticAlgorithm<<<GRID_SIZE, BLOCK_SIZE>>>(device);
		check(cudaThreadSynchronize());

		cudaMemcpy(host.population, device.population, POPULATION_SIZE*sizeof(metaChromosome),cudaMemcpyDeviceToHost);

		for (short i = 0; i < POPULATION_SIZE; i++){
			for(short j = 0; j < CHROMOSOME_SIZE; j++){
				std::cout << host.population[i].chromosome[j] << " ";
			}
			std::cout << host.population[i].distance << " " << host.population[i].fitness << std::endl;
		}

		cudaFree(device.population);
		check(cudaThreadSynchronize());

		std::cin.get();
	}catch(cudaError e){
		std::cout << "Oh crap: " << e << std::endl;
		std::cin.get();
	}
}
