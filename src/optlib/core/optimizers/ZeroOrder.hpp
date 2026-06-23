#pragma once

#include <cstddef>
#include <span>
#include <string_view>

#include <optlib/core/LinAlg.hpp>
#include <optlib/core/OptimizeResult.hpp>
#include <optlib/core/optimizers/FirstOrder.hpp>

namespace optlib {

enum class ZeroOrderMethod {
    NelderMead,
    Powell,
    CoordinateSearch,
};

struct ZeroOrderConfig {
    StopCriteria Criteria;
    bool StoreTrajectory = true;
    double InitialStep = 0.5;
    double LineSearchRadius = 1.0;
    double LineSearchTolerance = 1e-6;
    std::size_t LineSearchIterations = 80;
    double Reflection = 1.0;
    double Expansion = 2.0;
    double Contraction = 0.5;
    double Shrink = 0.5;
};

[[nodiscard]] ZeroOrderMethod ParseZeroOrderMethod(std::string_view methodName);

[[nodiscard]] OptimizeResult MinimizeZeroOrder(const ValueFunction& valueFunction,
                                               std::span<const double> x0,
                                               ZeroOrderMethod method,
                                               const ZeroOrderConfig& config = {});

[[nodiscard]] OptimizeResult MinimizeRosenbrockZeroOrder(std::span<const double> x0,
                                                         ZeroOrderMethod method,
                                                         const ZeroOrderConfig& config = {});

} // namespace optlib
