#ifndef TSP_H
#define TSP_H


#include "TSPParse.hpp"
#include <vector>
#include <fstream>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>
#include <boost/spirit/include/phoenix.hpp>


namespace spirit = boost::spirit;
namespace qi = boost::spirit::qi;
namespace phoenix = boost::phoenix;
namespace ascii = boost::spirit::ascii;

class TSP{
private:
	TSPList a;
public:
	TSP(char* filename){
		a = GetNodeListFromTSPFile(filename);
	}

	TSPList GetNodeListFromTSPFile(char* filename){
		using spirit::istream_iterator;
		using qi::space;
		using qi::space_type;

		std::ifstream file;
		TSPList result;
		TSPParse<istream_iterator> tsp;



		file.open(filename);
		istream_iterator begin(file);
		istream_iterator end;

		bool r = qi::parse(begin, end, tsp, result);

		return result;
	}

	TSPList list(){return a;}
	void list(TSPList list){a = list;}
};

#endif

