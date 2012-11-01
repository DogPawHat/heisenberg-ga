#ifndef TSPPARSE_H
#define TSPPARSE_H



#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>


namespace spirit = boost::spirit;
namespace qi = boost::spirit::qi;
namespace phoenix = boost::phoenix;
namespace ascii = boost::spirit::ascii;

typedef std::vector< std::vector<double> > TSPList ;

template <typename Iterator, typename Skipper>
struct TSPParse
: qi::grammar<Iterator, TSPList(), Skipper>
{
	TSPParse()
						: TSPParse::base_type(start, "tsp")
	{
		using qi::eps;
		using qi::int_;
		using qi::double_;
		using qi::_val;
		using qi::fail;
		using namespace qi::labels;
		using phoenix::push_back;
		using phoenix::val;
		using phoenix::construct;
		using ascii::char_;
		using qi::on_error;
		using qi::no_skip;
		using qi::eol;
		using ascii::space;

		start = (jibberish % eol) >> eol >> nodecoorddata[_val = _1];
		jibberish = name | type | comment | dimension | edgeweighttype;
		name = "NAME:" >> +char_("a-zA-Z_0-9()");
		type = "TYPE:" >> +char_("a-zA-Z_0-9()");
		comment = "COMMENT:" >> +char_("a-zA-Z_0-9()");
		dimension = "DIMENSION:" >> int_;
		edgeweighttype = "EDGE_WEIGHT_TYPE:" >> +char_("a-zA-Z_0-9()");
		nodecoorddata = "NODE_COORD_SECTION" >> eol >> (nodecoordentry % eol)[_val = _1] >> eol >> "EOF";
		nodecoordentry = int_ >> nodecoordpair[_val = _1];
		nodecoordpair = double_[push_back(_val, _1)] >> double_[push_back(_val, _1)];

		start.name("start");
		jibberish.name("jibberish");
		name.name("name");
		type.name("type");
		comment.name("comment");
		dimension.name("dimension");
		edgeweighttype.name("edgeweighttype");
		nodecoorddata.name("nodecoorddata");
		nodecoordentry.name("nodecoordentry");
		nodecoordpair.name("nodecoordpair");


		on_error<fail>
		(
				start
				, std::cout
				<< val("Error! Expecting ")
				<< _4                               // what failed?
				<< val(" here: \"")
				<< construct<std::string>(_3, _2)   // iterators to error-pos, end
				<< val("\"")
				<< '\n'
		);
	}
	qi::rule<Iterator, TSPList(), Skipper> start;
	qi::rule<Iterator, std::string(), Skipper> jibberish;
	qi::rule<Iterator, std::string(), Skipper> name;
	qi::rule<Iterator, std::string(), Skipper> type;
	qi::rule<Iterator, std::string(), Skipper> comment;
	qi::rule<Iterator, std::string(), Skipper> dimension;
	qi::rule<Iterator, std::string(), Skipper> edgeweighttype;
	qi::rule<Iterator, TSPList(), Skipper> nodecoorddata;
	qi::rule<Iterator, std::vector<double>(), Skipper> nodecoordentry;
	qi::rule<Iterator, std::vector<double>(), Skipper> nodecoordpair;
};

#endif
