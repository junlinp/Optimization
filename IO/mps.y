
%{
#include <string>
#include <iostream>
int yyerror(char*s);
int yylex(void);
%}

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
     | EndataSection
	  ;

EndataSection : SECTION_ENDATA;

NameSection: SECTION_NAME LINE { 
			std::cout << *($1) << "\n" << *($2)  << std::endl;
		    delete $1;
			delete $2;
			}
       ;

ObjSenseSection: SECTION_OBJSENSE IDENTIFIER {
			std::cout << *($1) << "\n" << *($2)  << std::endl;
		    delete $1;
			delete $2;
		    }
			   ;

RowSection : SECTION_ROW RowPart  { 
			std::cout << *($1) << "\n" << *($2)  << std::endl;
		    delete $1;
			delete $2;
			}
;

RowPart : RowPart RowItem { 
		$$ = new std::string(*$1 + " " + *$2); 
        delete $1;delete $2;}
        | RowItem {$$ = $1;}
        ;

RowItem : ROWTYPE IDENTIFIER { $$ = new std::string(*$1 + " " + *$2); 
		delete $1;delete $2;}
		;

ColumnsSection : SECTION_COLUMNS ColumnsPart;

ColumnsPart : ColumnsPart ColumnsItem 
		 | ColumnsItem;

ColumnsItem : IDENTIFIER IDENTIFIER NUMBER;

RhsSection : SECTION_RHS RhsPart;

RhsPart : RhsPart RhsItem
		| RhsItem;
RhsItem : IDENTIFIER IDENTIFIER NUMBER;

RangeSection : SECTION_RANGES;

BoundsSection : SECTION_BOUNDS BoundsPart;

BoundsPart : BoundsPart BoundsItem
		   | BoundsItem;
BoundsItem : IDENTIFIER IDENTIFIER IDENTIFIER NUMBER;

SosSection : SECTION_SOS;

EndataSection : SECTION_ENDATA;

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


int main() {
	yyparse();
}
