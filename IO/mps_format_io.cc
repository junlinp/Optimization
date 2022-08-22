#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "mps_format_io.h"


int yyparse();

int LoadMPSProblem(const std::string& path, Problem& problem) {
    extern FILE* yyin;
    extern Problem yyproblem;
    yyin = fopen(path.c_str(), "r");
    int res = yyparse();
    fclose(yyin);
    if (res == 0) {
        problem = yyproblem;
    }
    return res;
}