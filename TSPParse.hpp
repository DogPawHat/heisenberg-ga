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

using ascii::space_type;

template <typename Iterator>
struct TSPParse
: qi::grammar<Iterator, TSPList(), space_type>
{
	TSPParse()
						: TSPParse::base_type(start)
	{
		using qi::eps;
		using qi::int_;
		using qi::double_;
		using qi::_val;
		using qi::_1;
		using phoenix::push_back;
		using ascii::char_;

		start = (jibberish %'\n') >> nodecoorddata[_val = _1] >> "EOF";
		jibberish = +qi::char_("a-zA-Z_0-9:");
		nodecoorddata = "NODE_COORD_SECTION" >> (nodecoordentry % '\n')[_val = _1];
		nodecoordentry = int_ >> nodecoordpair[_val = _1];
		nodecoordpair = double_[push_back(_val, _1)] >> double_[push_back(_val, _1)];
	}
	qi::rule<Iterator, TSPList(), space_type> start;
	qi::rule<Iterator, std::string(), space_type> jibberish;
	qi::rule<Iterator, TSPList(), space_type> nodecoorddata;
	qi::rule<Iterator, std::vector<double>(), space_type> nodecoordentry;
	qi::rule<Iterator, std::vector<double>(), space_type> nodecoordpair;
};

#endif
