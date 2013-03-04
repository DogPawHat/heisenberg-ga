#include <cuda.h>
#include <math.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include "global_structs.h"


__device__ __forceinline__ void crossover(short* selectedMates, short* islandPopulation, deviceFields fields){
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
		parent1 = selectedMates[threadIdx.x*CHROMOSOME_SIZE];
		parent2 = selectedMates[(threadIdx.x+(BLOCK_SIZE/2))*CHROMOSOME_SIZE];
	}else{
		parent1 = selectedMates[threadIdx.x*CHROMOSOME_SIZE];
		parent2 = selectedMates[(threadIdx.x-(BLOCK_SIZE/2))*CHROMOSOME_SIZE];
	}

	dist1 = thrust::uniform_int_distribution<short>(0, 52);
	short point1 = dist1(rng);
	dist2 = thrust::uniform_int_distribution<short>(point1, 52)
	short point2 = dist2(rng);

	for(short i = point1; i <= point2; i++){
		offspring[i] = parent2[i];
	}

	for(int i = 0; i < point1; i++){
		for(int j = 0; j < (point2 - point1); j++){
			if(parent1[i] == offspring[j]){
				offspring[i] == parent2[i];
				goto a;
			}
		}
		offspring[i] == parent1[i];
	}

	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		islandPopulation[threadIdx*CHROMOSOME_SIZE+i] = offspring[i];
	}
}