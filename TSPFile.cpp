#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

typedef vector<float*> nodeList;

//Parses and stores data from a TSBLIB95 .tsp file (EUC_2D Node types)
class TSP{
	private:
		nodeList list;
	public:
		void setNodeListFromFile;
		float GetDistance;
}

float GetDistance()
{
	xd = x[i] - x[j];
	yd = y[i] - y[j];
	return (nint(sqrt(xd*xd + yd*yd ));
}


void SetNodeListFromTSPFile(string filename){
	ifstream TSPFile;
	stringstream currentLine;
	int linePos;
	int dumpInt;
	TSPFile.open(filename);
	while(currentLine.str != "NODE_COORD_SECTION"){
		getline(TSPFile, currentLine.str);
	}
	list = new nodeList;
	linePos = 1;
	while(currentLine.str != "EOF"){
		getline(TSPFile, currentLine);
		currentLine >> dumpInt;
		if(linePos == dumpInt){
			list.push_back(new float[2]) 
			currentLine >> list.back[0];
			currentLine >> list.back[1];
		}
	}
}