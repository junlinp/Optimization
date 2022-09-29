#/bin/bash

bison --defines="parser.h" --output parser.cc mps.yy
flex -o lex.cc mps.ll
g++ -std=c++17 lex.cc parser.cc mps_format_io.cc -lfl -c