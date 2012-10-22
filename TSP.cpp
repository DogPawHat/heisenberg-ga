#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>
using namespace std;

//Parses and stores data from a TSBLIB95 .tsp file (EUC_2D Node types)
class TSP{
	private:
		float** list;
	public:
		void SetNodeListFromTSPFile(string);
		float GetDistance(float*, float*);
		TSP(string);
};

TSP::TSP(string filename){
	this.list = GetNodeListFromTSPFile(filename);
}

void TSP::GetNodeListFromTSPFile(string filename){
	ifstream TSPFile;
	stringstream currentLine;
	int linePos;
	int listSize;
	float** list;


	TSPFile.open(filename);

	while(currentLine.str().find("DIMENSION:", 0) == string::npos){
		getline(TSPFile, currentLine.str());
	}
	currentLine.ignore(15, ':');
	currentLine >> listSize;
	list = new float[listSize][2];

	while(currentLine.str() != "EOF"){
		getline(TSPFile, currentLine.str());
		currentLine >> linePos;
		currentLine >> list[linePos][0];
		currentLine >> list[linePos][1];
	}
}

float TSP::GetDistance(float* x, float* y)
{
	float xd = x[i] - x[j];
	float yd = y[i] - y[j];
	return (nint(sqrt(xd*xd + yd*yd ));
}
