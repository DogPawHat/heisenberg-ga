#include <cuda.h>
#include <math.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include "global_structs.h"


/*
__device__ __noinline__ float randomRouletteBall(){
	thrust::minstd_rand rng;
	thrust::uniform_real_distribution<float> dist(0, 1);
	return dist(rng);
}

__device__ __noinline__ int* selection(int* matingPool, int* islandPopulation, deviceFields fields){
	__shared__ float fitnessValues[ISLAND_POPULATION_SIZE];
	__shared__ float totalFitnessValue;
	short selectedChromosome[CHROMOSOME_SIZE];
	int start = islandPopulation[threadIdx.x*CHROMOSOME_SIZE];

	for(int i = 1; i < CHROMOSOME_SIZE; i++){
		float xd = fields.TSPGraph[start+(i*2)] - fields.TSPGraph[start+(i*2)-2];
		float yd = fields.TSPGraph[start+(i*2)+1] - fields.TSPGraph[start+(i*2)-1];
		fitnessValues[threadIdx.x] = sqrtf(xd*xd + yd*yd);
		totalFitnessValue += sqrtf(xd*xd + yd*yd);
		__syncthreads();
	}

	fitnessValues[threadIdx.x] = fitnessValues[threadIdx.x]/totalFitnessValue;
	__syncthreads();

	float rouletteBall = randomRouletteBall();
	float diff = fdimf(fitnessValues[threadIdx.x], rouletteBall);
	memcpy(selectedChromosome, &islandPopulation[threadIdx.x*sizes.chromosomeSize], sizeof(int)*sizes.chromosomeSize);

	for(int i = 0; i < sizes.populationSize; i++){
		if(diff < fitnessValues[i] - rouletteBall){
			diff = fdimf(fitnessValues[threadIdx.x], rouletteBall);
			memcpy(selectedChromosome, &islandPopulation[i*sizes.chromosomeSize], sizeof(int)*sizes.chromosomeSize);
		}
		__syncthreads();
	}

	free(fitnessValues);
	return selectedChromosome;
}

__device__ __noinline__ void generation(short * islandPopulation, deviceFields fields){
	__shared__ short matingPool[MATING_POOL_SIZE]
}
*/

__global__ void runGeneticAlgorithm(deviceFields fields){
	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ short islandPopulation[ISLAND_POPULATION_SIZE];

/*
	for(short i = 0; i < CHROMOSOME_SIZE; i++){
		islandPopulation[threadIdx.x+i] = fields.population[gridIndex+i];
	}
	__syncthreads();
*/

	memcpy(&islandPopulation[threadIdx.x], &fields.population[gridIndex], CHROMOSOME_SIZE*sizeof(short));
	__syncthreads();


/*	for(short i = 0; i < CHROMOSOME_SIZE; i++){
		fields.population[gridIndex+i] = islandPopulation[threadIdx.x+i];
	}
	__syncthreads();
*/

	memcpy(&islandPopulation[threadIdx.x], &fields.population[gridIndex], CHROMOSOME_SIZE*sizeof(short));
	__syncthreads();
}
