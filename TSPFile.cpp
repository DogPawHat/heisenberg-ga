#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>
using namespace std;

typedef vector<float*> nodeList;

//Parses and stores data from a TSBLIB95 .tsp file (EUC_2D Node types)
class TSPFile{
	private:
		nodeList list;
	public:
		void SetNodeListFromTSPFile(string);
		float GetDistance(float*, float*);
};

void TSPFile::SetNodeListFromTSPFile(string filename){
	ifstream TSPFile;
	stringstream currentLine;
	int linePos;
	int dumpInt;
	TSPFile.open(filename);
	while(currentLine.str() != "NODE_COORD_SECTION"){
		getline(TSPFile, currentLine.str());
	}
	list
	linePos = 1;
	while(currentLine.str() != "EOF"){
		getline(TSPFile, currentLine.str());
		currentLine >> dumpInt;
		if(linePos == dumpInt){
			list.push_back(new float[2]);
			currentLine >> list.back[0];
			currentLine >> list.back[1];
		}
	}
}

float TSPFile::GetDistance(float* x, float* y)
{
	float xd = x[i] - x[j];
	float yd = y[i] - y[j];
	return (nint(sqrt(xd*xd + yd*yd ));
}
