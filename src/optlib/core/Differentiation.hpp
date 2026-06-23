#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <optlib/core/Dual.hpp>
#include <optlib/core/LinAlg.hpp>

namespace optlib {

enum class DifferentiationScheme {
    Forward,
    Central,
    FivePoint,
};

struct DifferentiationOptions {
    DifferentiationScheme Scheme = DifferentiationScheme::Central;
    double Step = 0.0;
};

using ObjectiveFunction = std::function<double(std::span<const double>)>;

[[nodiscard]] DifferentiationScheme ParseDifferentiationScheme(std::string_view schemeName);

[[nodiscard]] double DefaultStep(double coordinateValue, DifferentiationScheme scheme) noexcept;

void ComputeGradient(const ObjectiveFunction& function,
                     std::span<const double> x,
                     std::span<double> gradient,
                     const DifferentiationOptions& options = {});

[[nodiscard]] Vector ComputeGradient(const ObjectiveFunction& function,
                                     std::span<const double> x,
                                     const DifferentiationOptions& options = {});

template <class Function>
[[nodiscard]] Vector ComputeAutogradGradient(Function&& function, std::span<const double> x)
{
    auto dimension = x.size();
    if (dimension == 0) {
        throw std::invalid_argument("Autograd input dimension must be positive");
    }

    std::vector<Dual> dualValues;
    dualValues.reserve(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        dualValues.push_back(Dual::Variable(x[index], index, dimension));
    }

    auto result = function(std::span<const Dual>(dualValues.data(), dualValues.size()));
    return Vector(result.Gradient());
}

} // namespace optlib
