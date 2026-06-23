#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <string_view>

#include <optlib/core/LinAlg.hpp>
#include <optlib/core/OptimizeResult.hpp>
#include <optlib/core/StopCriteria.hpp>
#include <optlib/core/optimizers/FirstOrder.hpp>
#include <optlib/core/optimizers/LineSearch.hpp>

namespace optlib {

enum class SecondOrderMethod {
    Newton,
    BFGS,
    LBFGS,
};

struct SecondOrderConfig {
    StopCriteria Criteria;
    LineSearchConfig Search;
    std::size_t HistorySize = 10;
    double HessianDamping = 1e-6;
    bool StoreTrajectory = true;
};

using HessianFunction = std::function<void(std::span<const double>, Matrix&)>;

[[nodiscard]] SecondOrderMethod ParseSecondOrderMethod(std::string_view methodName);

[[nodiscard]] OptimizeResult MinimizeSecondOrder(const ValueFunction& valueFunction,
                                                 const GradientFunction& gradientFunction,
                                                 const HessianFunction& hessianFunction,
                                                 std::span<const double> x0,
                                                 SecondOrderMethod method,
                                                 const SecondOrderConfig& config = {});

[[nodiscard]] OptimizeResult MinimizeRosenbrockSecondOrder(std::span<const double> x0,
                                                           SecondOrderMethod method,
                                                           const SecondOrderConfig& config = {});

} // namespace optlib
