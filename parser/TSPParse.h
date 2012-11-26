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

template <typename Iterator>
struct TSPParse
: qi::grammar<Iterator, TSPList()>
{
	TSPParse(): TSPParse::base_type(start, "tsp")
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
		using ascii::space;

		start = (jibberish % no_skip['\n']) >> nodecoorddata[_val = _1] >> "EOF";
		jibberish = name | type | comment | dimension | edgeweighttype;
		name = "NAME: " >> +char_("a-zA-Z_0-9() ");
		type = "TYPE: " >> +char_("a-zA-Z_0-9() ");
		comment = "COMMENT:" >> +char_("a-zA-Z_0-9() ");
		dimension = "DIMESION:" >> int_;
		edgeweighttype = "EDGE_WEIGHT_TYPE:" >> +char_("a-zA-Z_0-9() ");
		nodecoorddata = "NODE_COORD_SECTION" >> (nodecoordentry % no_skip['\n'])[_val = _1];
		nodecoordentry = int_ >> space >> nodecoordpair[_val = _1];
		nodecoordpair = double_[push_back(_val, _1)] >> space >> double_[push_back(_val, _1)];

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

		debug(start);
		debug(jibberish);
		debug(name);
		debug(type);
		debug(comment);
		debug(dimension);
		debug(edgeweighttype);
		debug(nodecoorddata);
		debug(nodecoordentry);
		debug(nodecoordpair);

	}
	qi::rule<Iterator, TSPList()> start;
	qi::rule<Iterator, std::string()> jibberish;
	qi::rule<Iterator, std::string()> name;
	qi::rule<Iterator, std::string()> type;
	qi::rule<Iterator, std::string()> comment;
	qi::rule<Iterator, std::string()> dimension;
	qi::rule<Iterator, std::string()> edgeweighttype;
	qi::rule<Iterator, TSPList()> nodecoorddata;
	qi::rule<Iterator, std::vector<double>()> nodecoordentry;
	qi::rule<Iterator, std::vector<double>()> nodecoordpair;
};

#endif
