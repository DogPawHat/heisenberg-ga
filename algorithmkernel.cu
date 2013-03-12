#include <cuda.h>
#include <math.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include "global_structs.h"


__device__  void crossover(metaChromosome[], metaChromosome[], deviceFields);
__device__  void selection(metaChromosome[], metaChromosome[], deviceFields);

__device__ __forceinline__ void generation(short islandPopulation[], deviceFields fields){
	__shared__ metaChromosome selectedMates[ISLAND_POPULATION_SIZE];
	selection(selectedMates, islandPopulation, fields);
	crossover(selectedMates, islandPopulation, fields);
	__syncthreads();

	islandPopulation[threadIdx.x] = selectedMates[threadIdx.x];
}


__global__ void runGeneticAlgorithm(deviceFields fields){
	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ metaChromosome islandPopulation[ISLAND_POPULATION_SIZE];

	islandPopulation[threadIdx.x] = fields.population[gridIndex];
	__syncthreads();

	for(int i = 0; i < 1000; i++){
		generation(islandPopulation, fields);
		__syncthreads();
	}

	fields.population[gridIndex] = islandPopulation[threadIdx.x];
	__syncthreads();
}



/*Selection Functions*/

__device__  float randomRouletteBall(deviceFields fields){
	thrust::minstd_rand0 rng(fields.seeds[threadIdx.x + blockDim.x*blockIdx.x]);
	thrust::uniform_real_distribution<float> dist(0, 1);
	float result = dist(rng);
	return result;
}

__device__  void selection(metaChromosome selectedMates[], metaChromosome islandPopulation[], deviceFields fields){
	__shared__ float fitnessValues[ISLAND_POPULATION_SIZE];
	__shared__ float sumOfFitnessValues[ISLAND_POPULATION_SIZE];
	short start = threadIdx.x*CHROMOSOME_SIZE;

	for(short i = 1; i < CHROMOSOME_SIZE; i++){
		short j  = i - 1;
		float xi = fields.TSPGraph[2*islandPopulation.chromosome[i]];
		float xj = fields.TSPGraph[2*islandPopulation.chromosome[j]];
		float yi = fields.TSPGraph[2*islandPopulation.chromosome[i]+1];
		float yj = fields.TSPGraph[2*islandPopulation.chromosome[j]+1];
		float xd = fmaxf(xi, xj) - fminf(xi, xj);
		float yd = fmaxf(yi, yj) - fminf(yi, yj);
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


	

	float rouletteBall = 0;
	rouletteBall = randomRouletteBall(fields);
	float currentFitnessInterval = fitnessValues[0];
	selectedMates[threadIdx.x] = islandPopulation[0];

	for(short i = 1; i < ISLAND_POPULATION_SIZE; i++){
		if(rouletteBall > currentFitnessInterval){
			currentFitnessInterval += fitnessValues[i];
		}else{
			selectedMates[threadIdx.x] = islandPopulation[i]
			break;
		}
	}
}


/*Genetic Operators*/

__device__ __forceinline__ void crossover(metaChromosome selectedMates[], metaChromosome islandPopulation[], deviceFields fields){
	/*We need two different paths here beause each thread needs two parents to generate a single offspring.
	The first half of the block will take one parent from the first half of islandPopulation, while the second parent
	will come from the second half. This is reversed for the second half of the block. To reduce warp control divergence,
	block size shoud be a multiple of 2*warp size, 32 being the current value of warps in Fermi and Kepler GPU's*/
	
	short* parent1; //Points to the first element in the chromosome of parent1
	short* parent2;
	short point1;
	short point2;
	short size;
	short offspring[52];
	thrust::minstd_rand0 rng(fields.seeds[threadIdx.x+blockDim.x*blockIdx.x]);
	thrust::uniform_int_distribution<short> dist1;
	thrust::uniform_int_distribution<short> dist2;

	if(threadIdx.x < (BLOCK_SIZE/2)){
		parent1 = &selectedMates[threadIdx.x].chromosome;
		parent2 = &selectedMates[threadIdx.x+(BLOCK_SIZE/2)].chromosome;
	}else{
		parent1 = &selectedMates[threadIdx.x].chromosome;
		parent2 = &selectedMates[threadIdx.x-(BLOCK_SIZE/2)].chromosome;
	}

	dist1 = thrust::uniform_int_distribution<short>(0, 52);
	point1 = dist1(rng);
	dist2 = thrust::uniform_int_distribution<short>(point1, 52);
	point2 = dist2(rng);

	for(short i = point1; i <= point2; i++){
		offspring[i] = parent2[i];
	}

	for(int i = 0; i < point1; i++){
		for(int j = 0; j < (point2 - point1); j++){
			if(parent1[i] == offspring[j]){
				offspring[i] = parent2[i];
				goto a;
			}
		}
		offspring[i] = parent1[i];
		a:
	}

	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		islandPopulation[threadIdx.x].chromosome[i] = offspring[i];
	}
}
