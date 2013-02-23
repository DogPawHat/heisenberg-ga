#include <cuda.h>
#include <math.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include "global_structs.h"

__device__ __noinline__ float randomRouletteBall(){
	thrust::minstd_rand rng;
	thrust::uniform_real_distribution<float> dist(0, 1);
	return dist(rng);
}

__device__ __noinline__ void selection(short* selectedMates, short* islandPopulation, deviceFields fields){
	__shared__ float fitnessValues[ISLAND_POPULATION_SIZE];
	__shared__ float totalFitnessValue;
	__shared__ float worstFitnessValue;
	short start = threadIdx.x*CHROMOSOME_SIZE;

	for(int i = 1; i < CHROMOSOME_SIZE; i++){
		float xd = fdimf(fields.TSPGraph[islandPopulation[start+(i*2)]], fields.TSPGraph[islandPopulation[start+(i*2)-2]]);
		float yd = fdimf(fields.TSPGraph[islandPopulation[start+(i*2)+1]], fields.TSPGraph[islandPopulation[start+(i*2)-1]]);
		fitnessValues[threadIdx.x] += sqrtf(xd*xd + yd*yd);
		__syncthreads();
	}
	
	fitnessValues[threadIdx.x] = fitnessValues[threadIdx.x]/totalFitnessValue;
	fitnessValues[threadIdx.x] = 1 - fitnessValues[threadIdx.x];
	__syncthreads();

	float rouletteBall = randomRouletteBall();
	float currentFitnessInterval = fitnessValues[0];
	memcpy(&selectedMates[start], &islandPopulation[0], CHROMOSOME_SIZE*sizeof(short));

	for(short i = 1; i < ISLAND_POPULATION_SIZE; i++){
		if(fitnessValues[i] > currentFitnessInterval){
			currentFitnessInterval += fitnessValues[i];
		}else{
			memcpy(&selectedMates[start], &islandPopulation[i*CHROMOSOME_SIZE], CHROMOSOME_SIZE*sizeof(short));
			break;
		}
	}
}

__device__ __noinline__ void generation(short * islandPopulation, deviceFields fields){
	__shared__ short selectedMates[TOTAL_ISLAND_POPULATION_MEMORY_SIZE];
	selection(selectedMates, islandPopulation, fields);
	__syncthreads();

	memcpy(&islandPopulation[threadIdx.x], &selectedMates[threadIdx.x], CHROMOSOME_SIZE*sizeof(short));
}


__global__ void runGeneticAlgorithm(deviceFields fields){
	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ short islandPopulation[TOTAL_ISLAND_POPULATION_MEMORY_SIZE];

	memcpy(&islandPopulation[threadIdx.x], &fields.population[gridIndex], CHROMOSOME_SIZE*sizeof(short));
	__syncthreads();

	generation(islandPopulation, fields);
	__syncthreads();

	memcpy(&fields.population[gridIndex], &islandPopulation[threadIdx.x], CHROMOSOME_SIZE*sizeof(short));
	__syncthreads();
}
