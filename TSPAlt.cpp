#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/qi_stream.hpp>
#include <iostream>
#include <string>

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace phoenix = boost::phoenix;

using qi::double_;
using qi::_1;
using ascii::space;
using phoenix::ref;

struct TSPList{
	float*[2] a;
	TSPList(int size

	)
};

class TSP{
	public:
		TSP(char *);
		float** GetNodeListFromTSPFile(char*);
}

TSP::TSP(char* filename){
	this->list = GetNodeListFromTSPFile(filename);
};


float** TSP::GetNodeListFromTSPFile(char* filename){
	ifstream TSPFile;
	string match;
	string currentLine;
	int linePos;
	int listSize;
	float** list;


}

std::istream&
operator>>(istream& TSPStream, float**& TSPList){
	int linepos;
	linepos >> TSPList[linepos][0][1];
}
