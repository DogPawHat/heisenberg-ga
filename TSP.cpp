#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>
using namespace std;

//Parses and stores data from a TSBLIB95 .tsp file (EUC_2D Node types)
class TSP{
	public:
		float** list;
		TSP(char *);
		float** GetNodeListFromTSPFile(char*);
};

TSP::TSP(char* filename){
	string filenameS(filename);
	this->list = GetNodeListFromTSPFile(filenameS);
}

float** TSP::GetNodeListFromTSPFile(char* filename){
	ifstream TSPFile;
	stringstream currentLine;
	int linePos;
	int listSize;
	float** list;


	TSPFile.open(filename, ifstream::in);

	while(currentLine.str().find("DIMENSION:", 0) != string::npos){
		getline(TSPFile, currentLine.str());
	}
	currentLine.ignore(15, ':');
	currentLine >> listSize;
	list = new float* [listSize];

	while(currentLine.str() != "EOF"){
		getline(TSPFile, currentLine.str());
		currentLine >> linePos;
		list[linePos] = new float[2];
		currentLine >> list[linePos][0];
		currentLine >> list[linePos][1];
	}

	TSPFile.close();

	return list;
}
