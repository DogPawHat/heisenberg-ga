#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>
#include <boost/spirit/include/phoenix.hpp>


namespace spirit = boost::spirit;
namespace qi = boost::spirit::qi;
namespace phoenix = boost::phoenix;
namespace ascii = boost::spirit::ascii;

typedef vector<vector<double>> TSPList;
 
template <typename Iterator>
struct keys_and_values
	: qi::grammar<Iterator, TSPList()>
{
	using qi::eps;
	using qi::double_;
	using qi::_val;
	using qi::_1;
	using qi::_2;
	using phoenix::push_back;
	using ascii::char_;
	keys_and_values()
		: keys_and_values::base_type(start)
	{
		query =  (jibberish %'\n') >> nodecoorddata >> "EOF";
		jibberish = +qi::char_("a-zA-Z_0-9:");
		nodecoorddata = "NODE_COORD_SECTION" >> (nodecoordentry % '\n');
		nodecoordentry = int_ >> nodecoord[push_back(_val, _1);];
		nodecoordpair = nodecoord >> nodecoord;
		nodecoord = double_;
	}
	qi::rule<Iterator, TSPList()> query;
	qi::rule<Iterator, TSPList()> nodecoorddata
	qi::rule<Iterator, vector<double>()> nodecoordpair;
	qi::rule<Iterator, double> nodecoord;
};
