#ifndef TSP_H
#define TSP_H

#include <stdio.h>
#include <vector>
#include <fstream>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>
#include <boost/spirit/include/phoenix.hpp>


namespace spirit = boost::spirit;
namespace qi = boost::spirit::qi;
namespace phoenix = boost::phoenix;
namespace ascii = boost::spirit::ascii;

typedef std::vector<double[2]> TSPList ;


class TSP{
private:
	TSPList a;
public:
	TSP(char* filename){
		a = GetNodeListFromTSPFile(filename);
	}

	int parseDimensionSection(spirit::istream_iterator& first, spirit::istream_iterator last){

		using qi::int_;
		using qi::double_;
		using qi::string;
		using qi::_1;
		using phoenix::ref;
		using ascii::space;

		int size;

		qi::phrase_parse(first, last, (
				"NAME:" >> string >> "TYPE:" >> string >> "COMMENT:" >> string >> "DIMENSION:" >> int_[ref(size) = _1] >>"EDGE_WEIGHT_TYPE:" >> string
		),
				space
		);
		return size;
	}

	//This function parses the adjacency list in TSPLIB that represents the TSP graph.
	TSPList parseAdjacencyList(spirit::istream_iterator& first, spirit::istream_iterator last, int size){

		using qi::int_;
		using qi::double_;
		using qi::_1;
		using phoenix::ref;
		using phoenix::push_back;
		using ascii::space;

		int i; //Array index specified by the file.
		TSPList a(size); //Double array where the adjacency list will be stored

		//TODO: Exception Handling.
		qi::phrase_parse(first, last,
			(int_[ref(i) = _1] >> double_[ref(a[i][0]) = _1] >> double_[ref(a[i][1]) = _1]),
			space
		);

		return a;
	}

	TSPList GetNodeListFromTSPFile (char* filename){
		std::ifstream TSPFile;
		int size;

		TSPFile.open(filename);

		spirit::istream_iterator begin(TSPFile);
		spirit::istream_iterator end;

		size = parseDimensionSection(begin, end);
		return parseAdjacencyList(begin, end, size);
	}

	TSPList list(){return a;}
	void list(TSPList list){a = list;}
};

#endif