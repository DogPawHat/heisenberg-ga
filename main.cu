#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include "global_structs.h"
#include "berlin52.h"

__device__ deviceFields device;
deviceFields host;


__global__ void createRandomPermutation();
__global__ void createRandomSeeds(long seed);
__global__ void runGeneticAlgorithm();

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



		for(int i = 0; i < CHROMOSOME_SIZE; i++){
			host.source[i] = i;
			host.TSPGraph[i*2] = berlin52[i*2];
			host.TSPGraph[i*2 + 1] = berlin52[i*2 + 1];
		}

		check(cudaMemcpyToSymbol(device, &host, sizeof(deviceFields)));

		createRandomSeeds<<<GRID_SIZE, BLOCK_SIZE>>>(time(NULL));
		createRandomPermutation<<<GRID_SIZE, BLOCK_SIZE>>>();
		cudaDeviceSynchronize();

		runGeneticAlgorithm<<<GRID_SIZE, BLOCK_SIZE>>>();
		cudaDeviceSynchronize();

		check(cudaMemcpyFromSymbol(&host, device, sizeof(deviceFields)));

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
