
%{
#include <string>
#include <iostream>
#include "mps_format_io.h"

int yyerror(char*s);
int yylex(void);
extern FILE* yyin;

Problem yyproblem;

std::vector<std::string> equal_variable, less_variable, greater_variable;
std::string obj_variable;
%}
%require "3.2"

%union{
	double value;
	std::string* literature;
}



%token <literature>  SECTION_NAME
%token <literature> SECTION_ROW
%token <literature> SECTION_OBJSENSE
%token <literature> SECTION_COLUMNS
%token <literature> SECTION_RANGES
%token <literature> SECTION_BOUNDS
%token <literature> SECTION_SOS
%token <literature> SECTION_ENDATA
%token <literature> SECTION_RHS

%token <literature> LINE
%token <literature> IDENTIFIER
%token <value>  NUMBER
%token <literature> ROWTYPE

// In the BOUNDS section, bounds on the variables are specified.
// When bounds are not indicated, the default bounds ( 0 <= x < inf)
// are assumed. The code for indicating bound type is as follows:
// lower bound    b <= x ( < + inf)
%token <literature> BOUNDS_TYPE_LO
// upper bound    (0 <= ) x <= b
%token <literature> BOUNDS_TYPE_UP
// fixed variable  x = b
%token <literature> BOUNDS_TYPE_FX
// free variable   -inf < x < +inf
%token <literature> BOUNDS_TYPE_FR
// lower bound -inf -inf < x (<= 0)
%token <literature> BOUNDS_TYPE_MI
// upper bound +inf (0 <=) x < +inf
%token <literature> BOUNDS_TYPE_PL
// binary variable   x = 0 or 1
%token <literature> BOUNDS_TYPE_BV
// integer variable  b <= x (< +inf)
%token <literature> BOUNDS_TYPE_LI
// integer variable  (0 <=) x <= b
%token <literature> BOUNDS_TYPE_UI
// semi-cont variable x = 0 or l <= x <= b
// l is the lower bound on the variable
// If none set then defaults to 1
%token <literature> BOUNDS_TYPE_SC

%type <literature> RowItem
%type <literature> RowPart


%start input
%%
input : %empty
	 | NameSection input
     | ObjSenseSection input 
     | RowSection input
     | ColumnsSection input
     | RhsSection input
     | RangeSection input 
     | BoundsSection input
     | SosSection input
     | SECTION_ENDATA
	  ;

NameSection: SECTION_NAME LINE { 
		    delete $1;
			delete $2;
			}
       ;

ObjSenseSection: SECTION_OBJSENSE IDENTIFIER {
		    delete $1;
			delete $2;
		    }
			   ;

RowSection : SECTION_ROW RowPart  { 
		   yyproblem.equal_size = equal_variable.size();
		   yyproblem.less_size = less_variable.size();
		   yyproblem.greater_size = greater_variable.size();
		   int offset = 0;
           yyproblem.row_offset[obj_variable] = offset++;
		   for(auto str : equal_variable) {
				yyproblem.row_offset[str] = offset++;
		   }
		   for (auto str : less_variable) {
				yyproblem.row_offset[str] = offset++;
		   }
		   for (auto str : greater_variable) {
				yyproblem.row_offset[str] = offset++;
			}
			//std::printf("Equal %d, Less %d, Greater %d\n", yyproblem.equal_size, yyproblem.less_size, yyproblem.greater_size);
}
;

RowPart : RowPart RowItem
        | RowItem 
        ;

RowItem : ROWTYPE IDENTIFIER { 
		$$ = new std::string(*$1 + " " + *$2); 
        std::string row_type = *$1;
		if (*$1 == "E") {
			equal_variable.push_back(*$2);
		} else if (row_type == "G") {
			greater_variable.push_back(*$2);
		} else if (row_type == "L") {
			less_variable.push_back(*$2);
		} else {
			obj_variable = *$2;
		}
		delete $1;delete $2;
};

ColumnsSection : SECTION_COLUMNS ColumnsPart {
			std::printf("Total %lu nnz\n", yyproblem.row_column_value.size());
			   }

ColumnsPart : ColumnsPart ColumnsItem 
		 | ColumnsItem;

ColumnsItem : IDENTIFIER IDENTIFIER NUMBER {
			// column row value
			std::string column_name = *$1;
            std::string row_name = *$2;
double value = $3;
if (yyproblem.column_offset.find(column_name) == yyproblem.column_offset.end()) {
	yyproblem.column_offset[column_name] = yyproblem.column_offset.size();
}
yyproblem.row_column_value.push_back({row_name, column_name, value});
delete $1;
delete $2;
}

RhsSection : SECTION_RHS RhsPart {
		   std::printf("Total %lu RHS Item\n", yyproblem.rhs.size());
		   }

RhsPart : RhsPart RhsItem
		| RhsItem;
RhsItem : IDENTIFIER IDENTIFIER NUMBER {
		std::string row_name = *$2;
        double value = $3;
        yyproblem.rhs.emplace_back(row_name, value);
        delete $1;
        delete $2;
		}

RangeSection : SECTION_RANGES;

BoundsSection : SECTION_BOUNDS BoundsPart;

BoundsPart : BoundsPart BoundsItem
		   | BoundsItem;

BoundsItem : BOUNDS_TYPE_UP IDENTIFIER IDENTIFIER NUMBER {
		   std::string column_name = *$3;
double value = $4;

yyproblem.up_bounds.emplace_back(column_name, value);
delete $1;
delete $2;
delete $3;
}         | BOUNDS_TYPE_LO IDENTIFIER IDENTIFIER NUMBER {
    std::string column_name = *$3;
	double value = $4;
	yyproblem.lower_bounds.emplace_back(column_name, value);
	delete $1;
	delete $2;
	delete $3;
}

SosSection : SECTION_SOS;

%%


int yyerror(std::string s) {
    extern int yylineno;
    extern char* yytext;
    std::cerr << "ERROR : " << s << "at symbol \"" << yytext;
    std::cerr << "\" on line " << yylineno << std::endl;
	exit(1);
	return 0;
}

int yyerror(char* s) {
    yyerror(std::string(s));
	return 0;
}


/*
int main(int argc, char ** argv) {
    if (argc < 2) {
		printf("Usage: %s mps_file\n", argv[0]);
		return 0;
    }
    FILE* finput = fopen(argv[1], "r");
    yyin = finput;
	int res = yyparse();
    if (res == 0) {
		std::cout << "Parse Success" << std::endl;
    } else {
	    std::cout << "Parse Fails" << std::endl;
	}
}
*/
