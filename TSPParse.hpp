#ifndef TSPPARSE_H
#define TSPPARSE_H


#include <vector>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>
#include <boost/spirit/include/phoenix.hpp>


namespace spirit = boost::spirit;
namespace qi = boost::spirit::qi;
namespace phoenix = boost::phoenix;
namespace ascii = boost::spirit::ascii;

typedef std::vector< std::vector<double> > TSPList ;

using ascii::space_type;

template <typename Iterator>
struct TSP
: qi::grammar<Iterator, TSPList(), space_type>
{
	TSP()
						: TSP::base_type(start)
	{
		using qi::eps;
		using qi::double_;
		using qi::_val;
		using qi::_1;
		using qi::_2;
		using qi::_3;
		using phoenix::push_back;
		using ascii::char_;

		start =  eps[_val = new TSPList()] >> (jibberish %'\n') >> nodecoorddata[_val = _3] >> "EOF";
		jibberish = +qi::char_("a-zA-Z_0-9:");
		nodecoorddata = "NODE_COORD_SECTION" >> (nodecoordentry % '\n')[_val = _2];
		nodecoordentry = int_ >> nodecoordpair[_val = _2];
		nodecoordpair = nodecoord >> nodecoord;
		nodecoord = double_;
	}
	qi::rule<Iterator, TSPList(), space_type> start;
	qi::rule<Iterator, std::string(), space_type> jibberish;
	qi::rule<Iterator, TSPList(), space_type> nodecoorddata;
	qi::rule<Iterator, vector<double>, space_type> nodecoordentry;
	qi::rule<Iterator, vector<double>(), space_type> nodecoordpair;
	qi::rule<Iterator, double, space_type> nodecoord;
};

#endif
