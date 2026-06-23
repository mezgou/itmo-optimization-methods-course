#pragma once

#include <cstddef>
#include <span>
#include <string_view>

#include <optlib/core/LinAlg.hpp>
#include <optlib/core/optimizers/FirstOrder.hpp>

namespace optlib {

enum class LineSearchMethod {
    Armijo,
    StrongWolfe,
};

struct LineSearchConfig {
    LineSearchMethod Method = LineSearchMethod::StrongWolfe;
    double InitialStep = 1.0;
    double Reduction = 0.5;
    double C1 = 1e-4;
    double C2 = 0.9;
    double MinimumStep = 1e-12;
    std::size_t MaxIterations = 32;
};

[[nodiscard]] LineSearchMethod ParseLineSearchMethod(std::string_view methodName);

[[nodiscard]] double ArmijoLineSearch(const ValueFunction& valueFunction,
                                      std::span<const double> x,
                                      std::span<const double> direction,
                                      std::span<const double> gradient,
                                      double currentValue,
                                      const LineSearchConfig& config = {});

[[nodiscard]] double WolfeLineSearch(const ValueFunction& valueFunction,
                                     const GradientFunction& gradientFunction,
                                     std::span<const double> x,
                                     std::span<const double> direction,
                                     std::span<const double> gradient,
                                     double currentValue,
                                     const LineSearchConfig& config = {});

} // namespace optlib
