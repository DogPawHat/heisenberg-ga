#include <fstream>

using namespace std;

//Parses and stores data from a TSBLIB95 .tsp file (EUC_2D Node types)
class TSP{
	public:
		float** list;
		TSP(char *);
		float** GetNodeListFromTSPFile(char*);
		string MatchLineBeforeColon(istream TSPFile, string match){
};

TSP::TSP(char* filename){
	this->list = GetNodeListFromTSPFile(filename);
}

float** TSP::GetNodeListFromTSPFile(char* filename){
	ifstream TSPFile;
	string match;
	string currentLine;
	int linePos;
	int listSize;
	float** list;


	TSPFile.open(filename);

	match = "NAME:";

	TSP::MatchLineBeforeColon(TSPFile, match);

	match = "TYPE:";

	TSP::MatchLineBeforeColon(TSPFile, match);

	match = "COMMENT";

	TSP::MatchLineBeforeColon(TSPFile, match);

	match = "DIMENSION:";

	TSP::MatchLineBeforeColon(TSPFile, match);

	TSPFile >> listSize;

	match = "EDGE_WEIGHT_TYPE:";

	TSP::MatchLineBeforeColon(TSPFile, match);

	match = "NODE_COORD_SECTION";

	currentLineStream >> listSize;
	list = new float* [listSize];

	while(currentLine != "EOF"){
		getline(TSPFile, currentLine);
		currentLineStream.str(currentLine);
		currentLineStream >> linePos;
		list[linePos] = new float[2];
		currentLineStream >> list[linePos][0];
		currentLineStream >> list[linePos][1];
	}

	TSPFile.close();

	return list;
}

string TSP::MatchLineBeforeColon(istream TSPFile, string match){
	string currentline;
	while(currentLine != match){
				std::getline(TSPFile, currentLine, ':');
	}
	return currentline;
}
