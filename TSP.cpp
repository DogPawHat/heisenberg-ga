#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>


namespace spirit = boost::spirit;
namespace qi = boost::spirit::qi;
namespace phoenix = boost::phoenix;

using spirit::istream_iterator;
using qi::int_;
using qi::double_;
using qi::_1;
using phoenix::ref;



class TSP{
	private:
		double** a;
	public:
		TSP(char *);
		double** GetNodeListFromTSPFile(char*);
		int parseDimentionSection(istream_iterator&, istream_iterator);
		double** parseAdjacencyList(istream_iterator&, istream_iterator, int);
		double** list(){return a;}
		void list(double** list){a = list;}
};

TSP::TSP (char* filename){
	a = GetNodeListFromTSPFile(filename);
}


double** TSP::GetNodeListFromTSPFile (char* filename){
	std::ifstream TSPFile;
	int size;
	
	TSPFile.open(filename);
	
	istream_iterator begin(TSPFile);
	istream_iterator end;
	
	size = TSP::parseDimentionSection(begin, end);
	return TSP::parseAdjacencyList(begin, end, size);
}

//Parses data section.
int TSP::parseDimentionSection(istream_iterator& first, istream_iterator last){

	int a;

	qi::parse(first, last, (
			"NAME:" >> string_ >>
			"TYPE:" >> string_ >>
			"COMMENT:" >> string_ >>
			"DIMENSION:" >> int_[ref(a) = _1]
			"EDGE_WEIGHT_TYPE:" >> string_
		));

}


//This function parses the adjacency list in TSPLIB that represents the TSP graph.
double** TSP::parseAdjacencyList(istream_iterator& first, istream_iterator last, int size){

	
	int l; //Array index specified by the file.
	float a [size][2]; //Float array where the adjacency list will be stored
	
	//TODO: Exception Handling.
	qi::parse(first, last, (
			int_[ref(l) = _1] >>
			(double_[ref(a[l][0]) = _1] >>
			double_[ref(a[l][1]) = _1]) %
		)
	);
	
	return a;
}



