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
	short start = islandPopulation[threadIdx.x*CHROMOSOME_SIZE];

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
	memcpy(&selectedMates[start], &islandPopulation[start], CHROMOSOME_SIZE*sizeof(short));

	for(short i = 0; i < ISLAND_POPULATION_SIZE; i++){
		float newDiff = fdimf(fitnessValues[i], rouletteBall);
		if(newDiff < diff){
			diff = newDiff;
			memcpy(&selectedMates[start], &islandPopulation[i*CHROMOSOME_SIZE], CHROMOSOME_SIZE*sizeof(short));
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
