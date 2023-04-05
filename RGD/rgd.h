#ifndef RGB_RGD_H_
#define  RGB_RGD_H_
#include "so3_cost_function_interface.h"

bool rgd(const SO3CostFunctionInterface& cost_function, std::vector<SO3Manifold::Vector>* x_init);

#endif //  RGB_RGD_H_