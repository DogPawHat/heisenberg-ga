#ifndef TSPPARSE_H
#define TSPPARSE_H

#include <boost/spirit/include/qi.hpp>
#include "tsplist.h"

template <typename Iterator, typename Skipper>
struct TSPParse
: boost::spirit::qi::grammar<Iterator, Skipper, TSPList()>{
	TSPParse(): TSPParse::base_type(start, "tsp");
}

#endif