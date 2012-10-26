#include <boost/spirit/include/qi.hpp>
#include <iostream>
#include <string>

namespace spirit = boost::spirit;
namespace qi = spirit::qi;

class TSP{
	public:
		TSP(char *);
		double** GetNodeListFromTSPFile(char*);
}

TSP::TSP(char* filename){
	this->list = GetNodeListFromTSPFile(filename);
};


double** TSP::GetNodeListFromTSPFile(char* filename){
	std::ifstream TSPFile;
	
	TSPFile.open(filename);
	
	spirit::istream_iterator begin(TSPFile);
	spirit::istream_iterator end;
	
	TSP
	
	TSP::parseAjacenclyList(begin, end, size);
}

//Parses data section.
int TSP::parseDimentionSection(Iterator first&, Iterator last&){}


//This function parses the adjacency list in TSPLIB that represents the TSP graph.
double** TSP::parseAdjacencyList(Iterator& first, Iterator& last, int size){
	using qi::int_
	using qi::double_
	using qi::_1
	using phoenix::ref;
	
	int l; //Array index specified by the file.
	float a [size][2]; //Float array where the adjacentcy list will be stored
	
	//TODO: Exception Handling.
	qi::parse(first, last, (int_[ref(l) = _1] >> double_[ref(a[l][0])] >> double_[ref(a[l][1]])) %;
	
	return a;
}



