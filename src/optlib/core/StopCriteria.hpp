#pragma once

#include <cstddef>

namespace optlib {

struct StopCriteria {
    std::size_t MaxIterations = 10000;
    double GradientTolerance = 1e-6;
    double StepTolerance = 1e-12;
    double FunctionTolerance = 1e-12;
};

} // namespace optlib
