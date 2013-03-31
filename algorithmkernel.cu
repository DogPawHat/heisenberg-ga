#include <cuda.h>
#include <math.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include "rapidxml.hpp"
#include "rapidxml_utils.hpp"
#include "algorithm.cuh"


using thrust::random::minstd_rand0;
using thrust::random::uniform_int_distribution;

int main();
void check();
int chromosomeCheck();
void runGeneticAlgorithm();


__global__ void createRandomPermutation();
__global__ void createRandomSeeds(long seed);
__global__ void generation();
__device__ void migration(int);

__device__ void createNewSeed(long);

__device__ void selection();
__device__ void rouletteSelection();
__device__ float randomRouletteBall();
__device__ void fitnessEvauation(float[]);
__device__ void tournamentSelection();

__device__ void sort();

__device__ void crossover();
__device__ void crossoverOX(metaChromosome*, metaChromosome*);
__device__ void crossoverERO(metaChromosome*, metaChromosome*);

__device__ void mutation();




void check(cudaError call){
	if(call != cudaSuccess){
		throw &call;
	}
}

int chromosomeCheck(int chromosome[]){
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

int main(int argc, char* argv){
	try{

		for(int i = 0; i < CHROMOSOME_SIZE; i++){
			host.source[i] = i;
			host.TSPGraph[i*2] = berlin52[i*2];
			host.TSPGraph[i*2 + 1] = berlin52[i*2 + 1];
		}

		check(cudaMemcpyToSymbol(device, &host, sizeof(deviceFields)));

		createRandomSeeds<<<GRID_SIZE, BLOCK_SIZE>>>(time(NULL));
		createRandomPermutation<<<GRID_SIZE, BLOCK_SIZE>>>();
		check(cudaDeviceSynchronize());

		runGeneticAlgorithm();

		check(cudaMemcpyFromSymbol(&host, device, sizeof(deviceFields)));

		for (short i = 0; i < POPULATION_SIZE; i++){
			std::cout << '[' << chromosomeCheck(host.population[i].chromosome) << ']' << " ";
			for(short j = 0; j < CHROMOSOME_SIZE; j++){
				std::cout << host.population[i].chromosome[j] << " ";
			}
			std::cout << host.population[i].distance << std::endl;
		}
	}catch(cudaError * e){
		std::cout << "Oh crap: " << cudaGetErrorString(*e) << std::endl;
	}
}

void parseXMLInstance(char* filename){
	rapidxml::xml_document<> doc;
	rapidxml::file<> file(filename);
	doc.parse<0>(file.data());

	rapidxml::xml_node<>* graph = doc.first_node("graph");
	setChromosomeSize(rapidxml::count_children(graph));

	hostTSP.adjacencyMatrix= new double(CHROMOSOME_SIZE*CHROMOSOME_SIZE);
	cudaMalloc((void **) &(deviceTSP.adjacencyMatrix), CHROMOSOME_SIZE*CHROMOSOME_SIZE*sizeof(double));

	rapidxml::xml_node<>* vertex = graph->first_node("vertex");
	rapidxml::xml_node<>* edge;
	rapidxml::xml_attribute<>* cost;
	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		edge = vertex->first_node("edge");
		for(int j = 0; j < CHROMOSOME_SIZE; j++){
			double* currentCostHost = &(hostTSP.adjacencyMatrix[i*CHROMOSOME_SIZE+j]);
			double* currentCostDevice = &(deviceTSP.adjacencyMatrix[i*CHROMOSOME_SIZE+j]);

			if(i == j){
				*currentCostHost = 0;
				*currentCostDevice = 0;
			}else{
				cost = edge->first_attribute("cost");
				*currentCostHost = *(cost->value());
				*currentCostHost = *(cost->value());
			}
		}
	}
}

void runGeneticAlgorithm(){
	for(int i = 0; i < GENERATIONS; i++){
		generation<<<GRID_SIZE, BLOCK_SIZE>>>();
		check(cudaDeviceSynchronize());
	}
}


__global__ void createRandomPermutation(){
	int tempResult[CHROMOSOME_SIZE];
	int temp;
	int rand;
	int * chromosome = device.population[threadIdx.x+blockIdx.x*blockDim.x].chromosome;
//	short start = (threadIdx.x + blockIdx.x*blockDim.x)*CHROMOSOME_SIZE;

	minstd_rand0 rng(device.seeds[threadIdx.x+blockIdx.x*blockDim.x]);

	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		tempResult[i] = device.source[i];
	}

	for(int i = CHROMOSOME_SIZE-1; i >= 0; i--){
		uniform_int_distribution<int> dist(0,i);
		rand = dist(rng);
		temp = tempResult[rand];
		tempResult[rand] = tempResult[i];
		tempResult[i] = temp;
	}
	__syncthreads();

	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		chromosome[i] = tempResult[i];
	}
	device.population[threadIdx.x+blockIdx.x*blockDim.x].distanceCalculation(device.TSPGraph);
}

__global__ void createRandomSeeds(long seed){
	minstd_rand0 rng(seed*(threadIdx.x + blockIdx.x*blockDim.x)-34156);

	uniform_int_distribution<int> dist(0,RAND_MAX);
	device.seeds[threadIdx.x + blockDim.x*blockIdx.x]=dist(rng);
}

__global__ void generation(){
	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;
	islandPopulation[threadIdx.x] = device.population[gridIndex];
	__syncthreads();

	thrust::minstd_rand rng(device.seeds[gridIndex]);
	thrust::uniform_int_distribution<short> dist(1, 100);

	islandPopulation[threadIdx.x].distanceCalculation(device.TSPGraph);
	__syncthreads();

	selection();
	__syncthreads();

	if(dist(rng) < CROSSOVER_CHANCE){
		crossover();
		__syncthreads();
	}

	if(dist(rng) < MUTATION_CHANCE){
		mutation();
		__syncthreads();
	}

	islandPopulation[threadIdx.x].distanceCalculation(device.TSPGraph);
	__syncthreads();

	createNewSeed(device.seeds[gridIndex]);
	__syncthreads();

	sort();
	__syncthreads();

	islandPopulation[threadIdx.x].distanceCalculation(device.TSPGraph);
	__syncthreads();

	migration(gridIndex);
	__syncthreads();
}


/* Migration Functions */

__device__ void migration(int gridIndex){
	if(threadIdx.x < BLOCK_SIZE/2){
		device.population[gridIndex] = islandPopulation[threadIdx.x];
	}else if(blockIdx.x < GRID_SIZE - 1){
		device.population[gridIndex+BLOCK_SIZE] = islandPopulation[threadIdx.x-(BLOCK_SIZE/2)];
	}else{
		device.population[threadIdx.x] = islandPopulation[threadIdx.x-(BLOCK_SIZE/2)];
	}
}


/*Random Number Generator functions*/

__device__ void createNewSeed(long seed){
	thrust::minstd_rand rng(seed);

	thrust::uniform_int_distribution<int> dist(0,RAND_MAX);
	device.seeds[threadIdx.x + blockDim.x*blockIdx.x]=dist(rng);
}


/*Selection Functions*/

__device__ void selection(){
	tournamentSelection();
}

__device__ void rouletteSelection(){ //Dodge as fuck
	metaChromosome selectedIndividual;
	__shared__ float fitnessValues[ISLAND_POPULATION_SIZE];
	float rouletteBall;

	fitnessEvauation(fitnessValues);
	__syncthreads();

	rouletteBall = randomRouletteBall();

	for(short i = 1; i < ISLAND_POPULATION_SIZE; i++){
		if(rouletteBall < fitnessValues[i]){
			selectedIndividual = islandPopulation[i];
			break;
		}
	}

	islandPopulation[threadIdx.x] = selectedIndividual;
	__syncthreads();
}

__device__ float randomRouletteBall(){
	thrust::minstd_rand0 rng(device.seeds[threadIdx.x + blockDim.x*blockIdx.x]);
	thrust::uniform_real_distribution<float> dist(0, 1);
	float result = dist(rng);
	return result;
}

__device__ void fitnessEvauation(float fitnessValues[]){
	fitnessValues[threadIdx.x] = islandPopulation[ISLAND_POPULATION_SIZE-1].distance - islandPopulation[threadIdx.x].distance;
	__syncthreads();

	for(int stride = 1; stride < ISLAND_POPULATION_SIZE; stride *= 2){
		if(threadIdx.x + stride < ISLAND_POPULATION_SIZE){
			fitnessValues[threadIdx.x] += fitnessValues[threadIdx.x+stride];
			__syncthreads();
		}
	}

	fitnessValues[threadIdx.x] = fitnessValues[threadIdx.x]/fitnessValues[0];
}


__device__ void tournamentSelection(){
	int N = 5;
	metaChromosome tournamentChampion;
	metaChromosome tournamentChallenger;

	thrust::minstd_rand rng(device.seeds[threadIdx.x + blockIdx.x*blockDim.x]);
	thrust::uniform_int_distribution<short> dist(0, CHROMOSOME_SIZE-1);
	
	tournamentChampion = islandPopulation[threadIdx.x];

	for(int i = 0; i < N; i++){
		tournamentChallenger = islandPopulation[dist(rng)];
		if(tournamentChampion.distance > tournamentChallenger.distance){
			tournamentChampion = tournamentChallenger;
		}
	}

	__syncthreads();
	islandPopulation[threadIdx.x] = tournamentChampion;
}

/* Sorting Algorithms */

__device__ void sort(){
	for (int k = 2; k <= ISLAND_POPULATION_SIZE; k <<= 1){
		__syncthreads();
		for (int j=k>>1; j>0; j=j>>1){
			__syncthreads();
			unsigned int i = threadIdx.x; 
			int ixj = i^j;
		
			if ((ixj)>i){
				if ((i&k)==0 && islandPopulation[i].distance>islandPopulation[ixj].distanceCalculation(device.TSPGraph)){
					metaChromosome temp = islandPopulation[i];
					islandPopulation[i] = islandPopulation[ixj];
					islandPopulation[ixj] = temp;
					__syncthreads();
				}
				if ((i&k)!=0 && islandPopulation[i].distance<islandPopulation[ixj].distance){
					metaChromosome temp = islandPopulation[i];
					islandPopulation[i] = islandPopulation[ixj];
					islandPopulation[ixj] = temp;
					__syncthreads();
				}
			}
		}
	}
}


/*Genetic Operators*/

__device__ void mutation(){
	metaChromosome mutant = islandPopulation[threadIdx.x]; 
	thrust::minstd_rand0 rng(device.seeds[threadIdx.x+blockDim.x*blockIdx.x]);
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

	mutant.distanceCalculation(device.TSPGraph);
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
	thrust::minstd_rand0 rng(device.seeds[threadIdx.x+blockDim.x*blockIdx.x]);
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

	child.distanceCalculation(device.TSPGraph);
	islandPopulation[threadIdx.x] = child;
}

__device__ void crossoverERO(metaChromosome * parent1, metaChromosome * parent2){
	thrust::minstd_rand rng(device.seeds[threadIdx.x]);
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

	child.distanceCalculation(device.TSPGraph);
	islandPopulation[threadIdx.x] = child;
}

