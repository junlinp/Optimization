#ifndef IO_MPS_FORMAT_IO_H_
#define IO_MPS_FORMAT_IO_H_
#include <string>
#include <vector>
#include <map>
#include <tuple>

struct Problem {
	std::map<std::string, int> row_offset;	
	std::map<std::string, int> column_offset;
	int equal_size, less_size, greater_size;
	std::vector<std::pair<std::string, double>> rhs;
	std::vector<std::tuple<std::string,std::string, double>> row_column_value;
	std::vector<std::pair<std::string, double>> up_bounds, lower_bounds;
};

int LoadMPSProblem(const std::string& path, Problem& problem);

#endif // IO_MPS_FORMAT_IO_H_
