#include <cuda.h>
#include <math.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include "global_structs.h"

__device__ float randomRouletteBall(){
	thrust::minstd_rand rng;
	thrust::uniform_real_distribution<float> dist(0, 1);
	return dist(rng);
}

__device__ int* selection(int* matingPool, int* islandPopulation, deviceFields fields, const fieldSizes sizes){
	__shared__ float* fitnessValues;
	fitnessValues = (float*) malloc(sizeof(float)*sizes.populationSize);
	__shared__ float totalFitnessValue;
	int* selectedChromosome = (int*) malloc(sizeof(int)*sizes.populationSize);
	int start = islandPopulation[threadIdx.x*sizes.chromosomeSize];
	for(int i = 1; i < sizes.chromosomeSize; i++){
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
	return selectedChromosome;
}

__device__ void generation(int * islandPopulation, deviceFields fields, const fieldSizes sizes){
	__shared__ int * matingPool;
	matingPool = (int*) malloc(sizes.islandPopulationSize/10*sizeof(int));
	memcpy(&matingPool[threadIdx.x*sizes.chromosomeSize], selection(matingPool, islandPopulation, fields, sizes), sizeof(float)*sizes.chromosomeSize);
	__syncthreads();
	memcpy(&islandPopulation[threadIdx.x*sizes.chromosomeSize], &matingPool[threadIdx.x*sizes.chromosomeSize], sizes.chromosomeSize*sizeof(int));
}


__global__ void runGeneticAlgorithm(deviceFields fields, fieldSizes sizes){
	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ int  * islandPopulation;
	islandPopulation = (int*) malloc(sizes.islandPopulationSize*sizeof(int));
	memcpy(&islandPopulation[threadIdx.x*sizes.chromosomeSize], &fields.population[gridIndex*sizes.chromosomeSize], sizes.chromosomeSize*sizeof(int));
	generation(islandPopulation, fields, sizes);
	memcpy(&fields.population[gridIndex*sizes.chromosomeSize], &islandPopulation[blockDim.x*sizes.chromosomeSize], sizes.chromosomeSize*sizeof(int));
}
