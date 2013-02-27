#include <cuda.h>
#include <math.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include "global_structs.h"

__device__ __forceinline__ float randomRouletteBall(deviceFields fields){
	thrust::minstd_rand0 rng(fields.seeds[threadIdx.x + blockDim.x*blockIdx.x]);
	thrust::uniform_real_distribution<float> dist(0, 1);
	float result = dist(rng);
	return result;
}

__device__ __forceinline__ void selection(short* selectedMates, short* islandPopulation, deviceFields fields){
	__shared__ float fitnessValues[ISLAND_POPULATION_SIZE];
	__shared__ float sumOfFitnessValues[ISLAND_POPULATION_SIZE];
	short start = threadIdx.x*CHROMOSOME_SIZE;

	for(short i = 1; i < CHROMOSOME_SIZE; i++){
		short j  = i - 1;
		float xd = fdimf(fields.TSPGraph[2*islandPopulation[start+i]], fields.TSPGraph[2*islandPopulation[start+j]]);
		float yd = fdimf(fields.TSPGraph[2*islandPopulation[start+i]+1], fields.TSPGraph[2*islandPopulation[start+j]+1]);
		fitnessValues[threadIdx.x] += sqrtf(xd*xd + yd*yd);
		__syncthreads();
	}

	fitnessValues[threadIdx.x] = 1/fitnessValues[threadIdx.x];
	sumOfFitnessValues[threadIdx.x] = fitnessValues[threadIdx.x];
	__syncthreads();

	for(short stride = 1; stride < ISLAND_POPULATION_SIZE; stride *= 2){
		if(threadIdx.x + stride < ISLAND_POPULATION_SIZE){
			sumOfFitnessValues[threadIdx.x] += sumOfFitnessValues[threadIdx.x+stride];
		}
	}

	fitnessValues[threadIdx.x] = fitnessValues[threadIdx.x]/sumOfFitnessValues[0];


	

	float rouletteBall = randomRouletteBall(fields);
	float currentFitnessInterval = fitnessValues[0];
	for(short i = 0; i < CHROMOSOME_SIZE; i++){
		selectedMates[threadIdx.x*CHROMOSOME_SIZE+i] = islandPopulation[i];
	}

	for(short i = 1; i < ISLAND_POPULATION_SIZE; i++){
		if(fitnessValues[i] > currentFitnessInterval){
			currentFitnessInterval += fitnessValues[i];
		}else{
			for(short j = 0; j < CHROMOSOME_SIZE; j++){
				selectedMates[threadIdx.x*CHROMOSOME_SIZE+j] = islandPopulation[i*CHROMOSOME_SIZE+j];
			}
			break;
		}
	}
}

__device__ __forceinline__ void generation(short * islandPopulation, deviceFields fields){
	__shared__ short selectedMates[TOTAL_ISLAND_POPULATION_MEMORY_SIZE];
	selection(selectedMates, islandPopulation, fields);
	__syncthreads();

	for(short i = 0; i < CHROMOSOME_SIZE; i++){
		islandPopulation[threadIdx.x*CHROMOSOME_SIZE+i] = selectedMates[threadIdx.x*CHROMOSOME_SIZE+i];
	}
}


__global__ void runGeneticAlgorithm(deviceFields fields){
	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ short islandPopulation[TOTAL_ISLAND_POPULATION_MEMORY_SIZE];

	for(short i = 0; i < CHROMOSOME_SIZE; i++){
		islandPopulation[threadIdx.x*CHROMOSOME_SIZE+i] = fields.population[gridIndex*CHROMOSOME_SIZE+i];
	}
	__syncthreads();

	generation(islandPopulation, fields);
	__syncthreads();

	for(short i = 0; i < CHROMOSOME_SIZE; i++){
		fields.population[gridIndex*CHROMOSOME_SIZE+i] = islandPopulation[threadIdx.x*CHROMOSOME_SIZE+i];
	}
	__syncthreads();
}
