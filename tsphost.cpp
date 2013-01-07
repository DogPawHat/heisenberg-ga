#include "tsplist.h"
#include <vector>
#include <fstream>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>
#include <boost/spirit/include/phoenix.hpp>

namespace spirit = boost::spirit;
namespace qi = boost::spirit::qi;
namespace phoenix = boost::phoenix;
namespace ascii = boost::spirit::ascii;

template <typename Iterator, typename Skipper>
struct TSPParse
: qi::grammar<Iterator, Skipper, TSPList()>
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
		using ascii::blank;

		start = (jibberish % no_skip['\n']) >> '\n' >> nodecoorddata[_val = _1] >> '\n' >> "EOF";
		jibberish = name | type | comment | dimension | edgeweighttype;
		name = "NAME:" >> +char_("a-zA-Z_0-9()");
		type = "TYPE:" >> +char_("a-zA-Z_0-9()");
		comment = "COMMENT:" >> +char_("a-zA-Z_0-9()");
		dimension = "DIMENSION:">> int_;
		edgeweighttype = "EDGE_WEIGHT_TYPE:" >> +char_("a-zA-Z_0-9() ");
		nodecoorddata = "NODE_COORD_SECTION" >> char_('\n') >> nodecoordlist[_val = _1];
		nodecoordlist = nodecoordentry % '\n';
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
		nodecoordlist.name("nodecoordlist");
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

/*		debug(start);
		debug(jibberish);
		debug(name);
		debug(type);
		debug(comment);
		debug(dimension);
		debug(edgeweighttype);
		debug(nodecoorddata);
		debug(nodecoordlist);
		debug(nodecoordentry);
		debug(nodecoordpair);
*/
	}
	qi::rule<Iterator, Skipper, TSPList()> start;
	qi::rule<Iterator, Skipper, std::string()> jibberish;
	qi::rule<Iterator, Skipper, std::string()> name;
	qi::rule<Iterator, Skipper, std::string()> type;
	qi::rule<Iterator, Skipper, std::string()> comment;
	qi::rule<Iterator, Skipper, std::string()> dimension;
	qi::rule<Iterator, Skipper, std::string()> edgeweighttype;
	qi::rule<Iterator, Skipper, TSPList()> nodecoorddata;
	qi::rule<Iterator, Skipper, TSPList()> nodecoordlist;
	qi::rule<Iterator, Skipper, std::vector<double>()> nodecoordentry;
	qi::rule<Iterator, Skipper, std::vector<double>()> nodecoordpair;
};

TSPList GetNodeListFromTSPFile(char* filename){
	using spirit::istream_iterator;
	using qi::blank;
	using qi::blank_type;

	std::ifstream file;
	file.unsetf(std::ios_base::skipws);
	TSPList result;
	TSPParse<istream_iterator, blank_type> tsp;



	file.open(filename);
	istream_iterator begin(file);
	istream_iterator end;

	bool r = qi::phrase_parse(begin, end, tsp, blank, result);

	return result;
}