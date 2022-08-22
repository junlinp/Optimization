%{
#include <string>
#include "parser.hpp"

int yyerror(char *s);

//{MANTISSA} { return NUMBER;}
%}

NAME NAME
OBJSENSE OBJSENSE
ROWS ROWS
COLUMNS COLUMNS
RHS RHS
RANGES RANGES
BOUNDS BOUNDS
SOS SOS
ENDATA ENDATA
Variable [a-zA-Z][a-zA-z0-9_]+
ROW_TYPE [ENLG]
DIGITS [0-9]+
MANTISSA  (\+|\-)?{DIGITS}?(\.)?{DIGITS}?
Exponent  [DEde](\+|\-)?{DIGITS}
Num      {MANTISSA}{Exponent}?
Line ^\*.+[0-9]
%%
{NAME} { yylval.literature = new std::string(yytext); return SECTION_NAME;}
{ROWS} { yylval.literature = new std::string(yytext); return SECTION_ROW;} 
{OBJSENSE} { yylval.literature = new std::string(yytext); return SECTION_OBJSENSE;} 
{COLUMNS} { yylval.literature = new std::string(yytext); return SECTION_COLUMNS;}
{RHS} { yylval.literature = new std::string(yytext); return SECTION_RHS;}
{RANGES} { yylval.literature = new std::string(yytext); return SECTION_RANGES;}
{BOUNDS} { yylval.literature = new std::string(yytext); return SECTION_BOUNDS;}
{SOS} { yylval.literature = new std::string(yytext); return SECTION_SOS;}
{ENDATA} { yylval.literature = new std::string(yytext); return SECTION_ENDATA;}

{ROW_TYPE} { yylval.literature = new std::string(yytext); return ROWTYPE;}

LO { yylval.literature = new std::string(yytext); return BOUNDS_TYPE_LO;}
UP { yylval.literature = new std::string(yytext); return BOUNDS_TYPE_UP;}
FX { yylval.literature = new std::string(yytext); return BOUNDS_TYPE_FX;}
FR { yylval.literature = new std::string(yytext); return BOUNDS_TYPE_FR;}
MI { yylval.literature = new std::string(yytext); return BOUNDS_TYPE_MI;}
PL { yylval.literature = new std::string(yytext); return BOUNDS_TYPE_PL;}
BV { yylval.literature = new std::string(yytext); return BOUNDS_TYPE_BV;}
LI { yylval.literature = new std::string(yytext); return BOUNDS_TYPE_LI;}
UI { yylval.literature = new std::string(yytext); return BOUNDS_TYPE_UI;}
SC { yylval.literature = new std::string(yytext); return BOUNDS_TYPE_SC;}

{Line} { yylval.literature = new std::string(yytext); return LINE;}
{Variable} { yylval.literature = new std::string(yytext); return IDENTIFIER;}
{Num} { yylval.value = std::stod(yytext); return NUMBER;}
[\n] { yylineno++;}
. {}

%%

