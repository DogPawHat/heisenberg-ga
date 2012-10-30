#include <vector>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>
#include <boost/spirit/include/phoenix.hpp>


namespace spirit = boost::spirit;
namespace qi = boost::spirit::qi;
namespace phoenix = boost::phoenix;
namespace ascii = boost::spirit::ascii;

typedef std::vector< std::vector<double> > TSPList;

template <typename Iterator>
struct TSPFile
: qi::grammar<Iterator, TSPList()>
{
	TSPFile()
				: TSPFile::base_type(start)
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
	qi::rule<Iterator, TSPList()> start;
	qi::rule<Iterator, std::string()> jibberish;
	qi::rule<Iterator, TSPList() > nodecoorddata;
	qi::rule<Iterator, vector<double> > nodecoordentry;
	qi::rule<Iterator, vector<double>()> nodecoordpair;
	qi::rule<Iterator, double> nodecoord;
};
