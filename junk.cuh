/*
 * junk.cuh
 *
 *  Created on: 8 Apr 2013
 *      Author: ciaran
 *
 *      This header contains functions that are not functioning properly in normal code and
 *      will throw errors if incuded in the main algorithm. They are included here for completeness.
 *
 *
 */

#ifndef JUNK_CUH_
#define JUNK_CUH_

//Roullette Selection operator
__device__ void geneticAlgorithm::roulletteSelection(){
	sort();
	__syncthreads();

	int * selection = new int[CHROMOSOME_SIZE];

	for(int stride = 1; stride < ISLAND_POPULATION_SIZE; stride *= 2){
		if(threadIdx.x + stride < ISLAND_POPULATION_SIZE){
			islandPopulationDistance[threadIdx.x] += islandPopulationDistance[threadIdx.x+stride];
		}
	}

	islandPopulationDistance[threadIdx.x] = islandPopulationDistance[threadIdx.x] - islandPopulationDistance[ISLAND_POPULATION_SIZE];

	thrust::minstd_rand rng(seeds[threadIdx.x+blockIdx.x*blockDim.x]);
	thrust::uniform_real_distribution<double> dist(islandPopulationDistance[blockDim.x-1], islandPopulationDistance[0]);

	double roulletteBall = dist(rng);

	for(int i = BLOCK_SIZE-1; i >= 0; i--){
		if(roulletteBall > islandPopulationDistance[i] && roulletteBall < islandPopulationDistance[i-1]){
			for(int j = 0; j < CHROMOSOME_SIZE; j++){
				selection[j] = islandPopulationChromosome[i][j];
			}
			break;
		}
	}

	__syncthreads();
	for(int j = 0; j < CHROMOSOME_SIZE; j++){
		islandPopulationChromosome[threadIdx.x][j] = selection[j];
	}
	distanceCalculation();
	delete selection;
}

//Edge Recombanation Crossover
__device__ void geneticAlgorithm::crossoverERX(int * parent1, int * parent2){

	int ** edgeList = new int*[CHROMOSOME_SIZE];
	int * child = new int[CHROMOSOME_SIZE];

	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		edgeList[i] = new int[4];
		for(int j = 0; j < CHROMOSOME_SIZE; j++){
			for(int k = 0; k < CHROMOSOME_SIZE; k++){
				if(parent1[j] == i && parent2[k] == i){
					int xa, xb, ya, yb;

					if(j == 0){
						xa = parent1[CHROMOSOME_SIZE - 1];
						xb = parent1[j+1];
					}else if(j == (CHROMOSOME_SIZE-1)){
						xa = parent1[j - 1];
						xb = parent1[0];
					}else{
						xa = parent1[j - 1];
						xb = parent1[j+1];
					}

					if(k == 0){
						ya = parent2[CHROMOSOME_SIZE - 1];
						yb = parent2[k+1];
					}else if(k == (CHROMOSOME_SIZE-1)){
						ya = parent2[k-1];
						yb = parent2[0];
					}else{
						ya = parent2[k-1];
						yb = parent2[k+1];
					}

					edgeList[i][0] = xa;
					edgeList[i][1] = xb;
					if(xa != ya || xb != ya){
						edgeList[i][2] = ya;
					}else{
						edgeList[i][2] = CHROMOSOME_SIZE;
					}

					if(xa != yb || xb != yb){
						edgeList[i][3] = yb;
					}
					else
					{
						edgeList[i][3] = CHROMOSOME_SIZE;
					}
					break;
				}
			}
		}
	}

	int currentNode = parent2[0];


	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		if(currentNode < CHROMOSOME_SIZE && currentNode >= 0){
			child[i] = currentNode;

			for(int j = 0; j < CHROMOSOME_SIZE; j++){
				for(int k = 0; k < 4; k++){
					if(currentNode==edgeList[j][k]){
						edgeList[j][k] = CHROMOSOME_SIZE;
						break;
					}
				}
			}

			bool nonEmpty = false;
			for(int j = 0; j < 4; j++){
				if(edgeList[currentNode][j] < CHROMOSOME_SIZE){
					nonEmpty = true;
					break;
				}
			}


			if(nonEmpty){
				short currentListSize = 4;
				short listSize= 0;
				for(int j = 0; j < 4; j++){
					if(edgeList[currentNode][j] < CHROMOSOME_SIZE){
						listSize = 0;
						for(int k = 0; k < 4; k++){
							if(edgeList[edgeList[currentNode][j]][k] != CHROMOSOME_SIZE){
								listSize++;
							}
						}

						if(listSize <= currentListSize){
							currentNode = edgeList[currentNode][j];
							currentListSize = listSize;
						}
					}
				}
			}
			else if(i<CHROMOSOME_SIZE-1){
				int j = currentNode;
				int k = 4;
				do{
					if(j+1<CHROMOSOME_SIZE){
						j = j+1;
					}else{
						j = 0;
					}

					if(k+1<4){
						k = k+1;
					}else{
						k = 0;
					}


					currentNode= edgeList[i][j];
				}while(currentNode == CHROMOSOME_SIZE);
			}else{
				break;
			}
		}
	}

	__syncthreads();
	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		islandPopulationChromosome[threadIdx.x][i] = child[i];
	}
	__syncthreads();


	distanceCalculation();
	__syncthreads();
}

//Grefenstette greedy crossover
__device__ void geneticAlgorithm::crossoverGX(int * parent1, int * parent2){
	int * parent1Buffer = new int[CHROMOSOME_SIZE];
	int * parent2Buffer = new int[CHROMOSOME_SIZE];
	int * child = new int[CHROMOSOME_SIZE];
	thrust::minstd_rand0 rng(seeds[threadIdx.x+blockDim.x*blockIdx.x]);
	for(int i = 1; i < CHROMOSOME_SIZE; i++){
		parent1Buffer[i] = parent1[i];
		parent2Buffer[i] = parent2[i];
	}



	child[0] = parent1Buffer[0];
	for(int k = 0; k < CHROMOSOME_SIZE; k++){
		for(int i = k; i < CHROMOSOME_SIZE; i++){
			for(int j = k; j < CHROMOSOME_SIZE; j++){
				if(child[k] == parent1Buffer[i-1] && child[k] == parent2Buffer[j-1]){
					if(distanceBetweenTwoCities(child[k], parent1Buffer[i]) < distanceBetweenTwoCities(child[k], parent2Buffer[i])){
					child[k+1] = parent1Buffer[i];
					}else{
					child[k+1] = parent2Buffer[i];
					}

					int tempA = parent1Buffer[k];
					int tempB = parent2Buffer[k];
					parent1Buffer[k] = parent1Buffer[i-1];
					parent2Buffer[k] = parent2Buffer[j-1];
					parent1Buffer[i-1] = tempA;
					parent2Buffer[j-1] = tempB;
					break;
				}
			}
		}
	}

	distanceCalculation();
	delete child;
}
#endif /* JUNK_CUH_ */
