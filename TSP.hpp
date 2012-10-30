#ifndef TSP_H
#define TSP_H

#include <stdio.h>
#include <vector>
#include <fstream>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>
#include <boost/spirit/include/phoenix.hpp>


namespace spirit = boost::spirit;
namespace qi = boost::spirit::qi;
namespace phoenix = boost::phoenix;
namespace ascii = boost::spirit::ascii;

typedef std::vector<double[2]> TSPList ;


template <typename Iterator>
struct TSPFile;

#endif
