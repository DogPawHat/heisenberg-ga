#ifndef TSP_H
#define TSP_H

#include "TSPParse.h"
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
	std::vector< std::vector<double> > a;
public:
	TSP(char* filename){
		a = GetNodeListFromTSPFile(filename);
	}

	std::vector< std::vector<double> > GetNodeListFromTSPFile(char* filename){
		using spirit::istream_iterator;
		using qi::space;
		using qi::space_type;

		std::ifstream file;
		file.unsetf(std::ios_base::skipws);
		std::vector< std::vector<double> > result;
		TSPParse<istream_iterator> tsp;



		file.open(filename);
		istream_iterator begin(file);
		istream_iterator end;

		bool r = qi::parse(begin, end, tsp, result);

		return result;
	}

	std::vector< std::vector<double> > list(){return a;}
	void list(std::vector< std::vector<double> > list){a = list;}
};
#endif
