#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include "global_structs.h"
#include "berlin52.h"

__global__ void createRandomPermutation(deviceFields*);
__global__ void createRandomSeeds(deviceFields*, long seed);
__global__ void runGeneticAlgorithm(deviceFields*);

void check(cudaError call){
	if(call != cudaSuccess){
		throw &call;
	}
}

int chromosomeCheck(short chromosome[]){
	int k;
	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		k = 0;
		for(int j = 0; j < CHROMOSOME_SIZE; j++){
			if(chromosome[j] == i){
				k++;
			}else if(chromosome[j] > CHROMOSOME_SIZE || chromosome[j] < 0){
				return 1;
			}
		}
		if(k != 1){
			return 1;
		}
	}
	return 0;
}

int main(){
	try{
		deviceFields * device;
		hostFields host;

		check(cudaMalloc((void**)&device, sizeof(deviceFields)));

		cudaMemcpy(device->TSPGraph, berlin52, 2*CHROMOSOME_SIZE*sizeof(float), cudaMemcpyHostToDevice);

		for(int i = 0; i < CHROMOSOME_SIZE; i++){
			host.source[i] = i;
		}

		check(cudaMemcpy(device->source, host.source, CHROMOSOME_SIZE*sizeof(short), cudaMemcpyHostToDevice));

		createRandomSeeds<<<GRID_SIZE, BLOCK_SIZE>>>(device, time(NULL));
		createRandomPermutation<<<GRID_SIZE, BLOCK_SIZE>>>(device);
		check(cudaDeviceSynchronize());

		runGeneticAlgorithm<<<GRID_SIZE, BLOCK_SIZE>>>(device);
		check(cudaDeviceSynchronize());

		check(cudaMemcpy(host.population, device->population, POPULATION_SIZE*sizeof(metaChromosome),cudaMemcpyDeviceToHost));

		for (short i = 0; i < POPULATION_SIZE; i++){
			std::cout << '[' << chromosomeCheck(host.population[i].chromosome) << ']' << " ";
			for(short j = 0; j < CHROMOSOME_SIZE; j++){
				std::cout << host.population[i].chromosome[j] << " ";
			}
			std::cout << host.population[i].distance << " " << host.population[i].fitness << std::endl;
		}
	}catch(cudaError * e){
		std::cout << "Oh crap: " << cudaGetErrorString(*e) << std::endl;
	}
}
