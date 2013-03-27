#include <cuda.h>
#include <math.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include "global_structs.h"


__shared__ metaChromosome islandPopulation[ISLAND_POPULATION_SIZE];
__device__ deviceFields * fields;

__device__ void generation();

__device__ void migration(int);

__device__ void createNewSeed(long);

__device__ void selection();
__device__ float randomRouletteBall();
__device__ void fitnessEvauation();

__device__ void sort();
__device__ void sortingBlock(int);
__device__ void mergingBlock(int, int, int, int, bool);
__device__ void compareAndSwap(int, bool);

__device__ void crossover();
__device__ void crossoverOX(metaChromosome*, metaChromosome*);
__device__ void crossoverERO(metaChromosome*, metaChromosome*);

__device__ void mutation();

__global__ void runGeneticAlgorithm(deviceFields* device){
	fields = device;
	for(int i = 0; i < GENERATIONS; i++){
		generation();
		__syncthreads();
	}
}

__device__ void generation(){
	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;
	migration(gridIndex);
//	islandPopulation[threadIdx.x] = fields.population[gridIndex];
	__syncthreads();

	thrust::minstd_rand rng(fields->seeds[gridIndex]);
	thrust::uniform_int_distribution<short> dist(1, 100);

	if(dist(rng) < CROSSOVER_CHANCE){
		crossover();
	}

	if(dist(rng) < MUTATION_CHANCE){
		mutation();
	}
	

	selection();
	__syncthreads();

	createNewSeed(fields->seeds[gridIndex]);
	__syncthreads();

//	sort();
//	__syncthreads();

	fields->population[gridIndex] = islandPopulation[threadIdx.x];
	__syncthreads();
}


/* Migration Functions */

__device__ void migration(int gridIndex){
	if(threadIdx.x < BLOCK_SIZE/2){
		islandPopulation[threadIdx.x] = fields->population[gridIndex];
	}else if(blockIdx.x < GRID_SIZE - 1){
		islandPopulation[threadIdx.x] = fields->population[gridIndex + BLOCK_SIZE/2];
	}else{
		islandPopulation[threadIdx.x] = fields->population[threadIdx.x - BLOCK_SIZE/2];
	}
}


/*Random Number Generator functions*/

__device__ void createNewSeed(long seed){
	thrust::minstd_rand rng(seed);

	thrust::uniform_int_distribution<int> dist(0,RAND_MAX);
	fields->seeds[threadIdx.x + blockDim.x*blockIdx.x]=dist(rng);
}


/*Selection Functions*/

__device__ void selection(){
	__shared__ metaChromosome selectedPopulation[ISLAND_POPULATION_SIZE];
	float rouletteBall;
	float currentFitnessInterval;

	islandPopulation[threadIdx.x].distanceCalculation(fields->TSPGraph);
	__syncthreads();
	fitnessEvauation();
	__syncthreads();

	rouletteBall = randomRouletteBall();
	currentFitnessInterval = islandPopulation[0].fitness;
	selectedPopulation[threadIdx.x] = islandPopulation[0];

	for(short i = 1; i < ISLAND_POPULATION_SIZE; i++){
		if(rouletteBall > currentFitnessInterval){
			currentFitnessInterval += islandPopulation[i].fitness;
		}else{
			selectedPopulation[threadIdx.x] = islandPopulation[i];
			break;
		}
	}

	islandPopulation[threadIdx.x] = selectedPopulation[threadIdx.x];
	__syncthreads();
}

__device__ float randomRouletteBall(){
	thrust::minstd_rand0 rng(fields->seeds[threadIdx.x + blockDim.x*blockIdx.x]);
	thrust::uniform_real_distribution<float> dist(0, 1);
	float result = dist(rng);
	return result;
}

__device__ void fitnessEvauation(){
	__shared__ float sumOfFitnessValues[ISLAND_POPULATION_SIZE];
	islandPopulation[threadIdx.x].fitness = 1/islandPopulation[threadIdx.x].distance;
	sumOfFitnessValues[threadIdx.x] = islandPopulation[threadIdx.x].fitness;
	__syncthreads();

	for(short stride = 1; stride < ISLAND_POPULATION_SIZE; stride *= 2){
		if(threadIdx.x + stride < ISLAND_POPULATION_SIZE){
			sumOfFitnessValues[threadIdx.x] += sumOfFitnessValues[threadIdx.x+stride];
			__syncthreads();
		}
	}

	islandPopulation[threadIdx.x].fitness = islandPopulation[threadIdx.x].fitness/sumOfFitnessValues[0];
}


/* Sorting Algorithms */

__device__ void sort(){
	for(int i = 2; i <= ISLAND_POPULATION_SIZE; i++){
		sortingBlock(i);
	}

}

__device__ void sortingBlock(int sortBlockDim){
	int sortBlockIdx = threadIdx.x/sortBlockDim;
	int sortThreadIdx = threadIdx.x%sortBlockDim;
	bool ascending = (sortBlockIdx%2==0);

	for(int i = sortBlockDim; i > 2; i++){
		mergingBlock(sortBlockDim, sortBlockIdx, sortThreadIdx, i, ascending);
	}
}

__device__ void mergingBlock(int sortBlockDim, int sortBlockIdx, int sortThreadIdx, int mergingBlockDim, bool ascending){
	int mergingBlockIdx = sortBlockIdx/mergingBlockDim;
	int mergingThreadIdx = sortBlockIdx%mergingBlockDim;

	if(mergingThreadIdx < mergingBlockDim/2){
		compareAndSwap(threadIdx.x+(mergingBlockDim/2), ascending);
	}
	__syncthreads();
}

__device__ void compareAndSwap(int i, bool ascending)
{
	if(ascending==(islandPopulation[threadIdx.x].distanceCalculation(fields->TSPGraph)>islandPopulation[i].distanceCalculation(fields->TSPGraph))){
		metaChromosome t=islandPopulation[threadIdx.x];
		islandPopulation[threadIdx.x]=islandPopulation[i];
		islandPopulation[i]=t;
	}
}


/*Genetic Operators*/

__device__ void mutation(){
	metaChromosome mutant = islandPopulation[threadIdx.x]; 
	thrust::minstd_rand0 rng(fields->seeds[threadIdx.x+blockDim.x*blockIdx.x]);
	thrust::uniform_int_distribution<short> dist1(0, 10);
	thrust::uniform_int_distribution<short> dist2(0, CHROMOSOME_SIZE-1);
	short numOfSwaps = dist1(rng);
	short swapPoint1;
	short swapPoint2;
	short temp;

	for(int i = 0; i < numOfSwaps; i++){
		swapPoint1 = dist2(rng);
		swapPoint2 = dist2(rng);
		temp = mutant.chromosome[swapPoint1];
		mutant.chromosome[swapPoint1] = mutant.chromosome[swapPoint2];
		mutant.chromosome[swapPoint2] = temp;
	}

	mutant.distanceCalculation(fields->TSPGraph);
	islandPopulation[threadIdx.x] = mutant;
}

__device__ void crossover(){
	metaChromosome * parent1;
	metaChromosome * parent2;
	
	if(threadIdx.x < (BLOCK_SIZE/2)){
		parent1 = &islandPopulation[threadIdx.x];
		parent2 = &islandPopulation[threadIdx.x+(BLOCK_SIZE/2)];
	}else{
		parent1 = &islandPopulation[threadIdx.x];
		parent2 = &islandPopulation[threadIdx.x-(BLOCK_SIZE/2)];
	}

	crossoverOX(parent1, parent2);
}

__device__ void crossoverOX(metaChromosome * parent1, metaChromosome * parent2){
	/*We need two different paths here beause each thread needs two parents to generate a single offspring.
	The first half of the block will take one parent from the first half of islandPopulation, while the second parent
	will come from the second half. This is reversed for the second half of the block. To reduce warp control divergence,
	block size shoud be a multiple of 2*warp size, 32 being the current value of warps in Fermi and Kepler GPU's*/
	
	short point1;
	short point2;
	metaChromosome child;
	thrust::minstd_rand0 rng(fields->seeds[threadIdx.x+blockDim.x*blockIdx.x]);
	thrust::uniform_int_distribution<short> dist1;
	thrust::uniform_int_distribution<short> dist2;

	dist1 = thrust::uniform_int_distribution<short>(0, CHROMOSOME_SIZE-1);
	point1 = dist1(rng);
	dist2 = thrust::uniform_int_distribution<short>(point1, CHROMOSOME_SIZE-1);
	point2 = dist2(rng);

	for(short i = point1; i <= point2; i++){
		child.chromosome[i] = parent1->chromosome[i];
	}


	short position = 0;
	for(short i = 0; i < CHROMOSOME_SIZE; i++){
		if(!(position < point1 || position > point2)){
			position = point2 + 1;
		}
		if(!(position < CHROMOSOME_SIZE)){
			break;
		}


		bool nonDuplicate = true;
		for(short j = point1; j <= point2; j++){
			if(child.chromosome[j] == parent2->chromosome[i]){
				nonDuplicate = false;
				break;
			}
		}
		if(nonDuplicate == true){
			child.chromosome[position] = parent2->chromosome[i];
			position++;
		}
	}

	child.distanceCalculation(fields->TSPGraph);
	islandPopulation[threadIdx.x] = child;
}

__device__ void crossoverERO(metaChromosome * parent1, metaChromosome * parent2){
	thrust::minstd_rand rng(fields->seeds[threadIdx.x]);
	thrust::uniform_int_distribution<short> dist(0, (CHROMOSOME_SIZE-1));
	unsigned short unionAjacency[CHROMOSOME_SIZE][4];
	metaChromosome child;
	unsigned short currentAvailable[CHROMOSOME_SIZE];
	unsigned short currentNode= dist(rng);

	for(unsigned short i = 0; i < CHROMOSOME_SIZE; i++){
		currentAvailable[i] = i;
	}

	currentAvailable[currentNode]= CHROMOSOME_SIZE + 1;

	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		for(int j = 0; j < CHROMOSOME_SIZE; j++){
			for(int k = 0; k < CHROMOSOME_SIZE; k++){
				if(parent1->chromosome[j] == i && parent2->chromosome[k] == i){
					unsigned short xa, xb, ya, yb;

					switch(j){
					case 0:
						xa = parent1->chromosome[CHROMOSOME_SIZE - 1];
						xb = parent1->chromosome[j+1];
						break;
					case CHROMOSOME_SIZE-1:
						xa = parent1->chromosome[j - 1];
						xb = parent1->chromosome[0];
						break;
					default:
						xa = parent1->chromosome[j - 1];
						xb = parent1->chromosome[j+1];
						break;
					}

					switch(k){
					case 0:
						ya = parent2->chromosome[CHROMOSOME_SIZE - 1];
						yb = parent2->chromosome[k+1];
						break;
					case CHROMOSOME_SIZE-1:
						ya = parent2->chromosome[k-1];
						yb = parent2->chromosome[0];
						break;
					default:
						ya = parent2->chromosome[k-1];
						yb = parent2->chromosome[k+1];
						break;
					}
					if(xa <= CHROMOSOME_SIZE && xb <= CHROMOSOME_SIZE && ya <= CHROMOSOME_SIZE && ya <= CHROMOSOME_SIZE){
						unionAjacency[i][0] = xa;
						unionAjacency[i][1] = xb;
						if(xa != ya || xb != ya){
							unionAjacency[i][2] = ya;
						}else{
							unionAjacency[i][2] = CHROMOSOME_SIZE+1;
						}

						if(xa != yb || xb != yb){
							unionAjacency[i][3] = yb;
						}
						else
						{
							unionAjacency[i][3] = CHROMOSOME_SIZE+1;
						}
						break;
					}
				}
			}
		}
	}


	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		if(currentNode < CHROMOSOME_SIZE && currentNode >= 0){
			child.chromosome[i] = currentNode;
			currentAvailable[currentNode]= CHROMOSOME_SIZE + 1;

			for(int j = 0; j < CHROMOSOME_SIZE; j++){
				for(int k = 0; k < 4; k++){
					if(currentNode==unionAjacency[j][k]){
						unionAjacency[j][k] = CHROMOSOME_SIZE+1;
						break;
					}
				}
			}

			bool nonEmpty = false;
			for(int j = 0; j < 4; j++){
				if(unionAjacency[currentNode][j] < CHROMOSOME_SIZE){
					nonEmpty = true;
					break;
				}
			}


			if(nonEmpty == true){
				short currentListSize = 4;
				short listSize= 0;
				for(int j = 0; j < 4; j++){
					if(unionAjacency[currentNode][j] < CHROMOSOME_SIZE){
						listSize = 0;
						for(int k = 0; k < 4; k++){
							if(unionAjacency[unionAjacency[currentNode][j]][k] != CHROMOSOME_SIZE+1){
								listSize++;
							}
						}

						if(listSize <= currentListSize){
							currentNode = unionAjacency[currentNode][j];
							currentListSize = listSize;
						}
					}
				}
			}
			else{
				do{
					currentNode= dist(rng);
				}while(currentAvailable[currentNode]== CHROMOSOME_SIZE + 1);
			}
		}
	}

	child.distanceCalculation(fields->TSPGraph);
	islandPopulation[threadIdx.x] = child;
}

