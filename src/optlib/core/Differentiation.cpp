#include <optlib/core/Differentiation.hpp>

#include <algorithm>
#include <string_view>

namespace optlib {

[[nodiscard]] DifferentiationScheme ParseDifferentiationScheme(std::string_view schemeName)
{
    if (schemeName == "forward" || schemeName == "Forward") {
        return DifferentiationScheme::Forward;
    }
    if (schemeName == "central" || schemeName == "Central") {
        return DifferentiationScheme::Central;
    }
    if (schemeName == "five-point" || schemeName == "five_point" || schemeName == "FivePoint") {
        return DifferentiationScheme::FivePoint;
    }
    throw std::invalid_argument("Unknown differentiation scheme");
}

[[nodiscard]] double DefaultStep(double coordinateValue, DifferentiationScheme scheme) noexcept
{
    constexpr double EPS = std::numeric_limits<double>::epsilon();
    auto scale = std::max(1.0, std::abs(coordinateValue));
    switch (scheme) {
    case DifferentiationScheme::Forward:
        return std::sqrt(EPS) * scale;
    case DifferentiationScheme::Central:
        return std::cbrt(EPS) * scale;
    case DifferentiationScheme::FivePoint:
        return std::pow(EPS, 0.2) * scale;
    }
    return std::cbrt(EPS) * scale;
}

void ComputeGradient(const ObjectiveFunction& function,
                     std::span<const double> x,
                     std::span<double> gradient,
                     const DifferentiationOptions& options)
{
    if (x.size() != gradient.size()) {
        throw std::invalid_argument("Gradient output size does not match input dimension");
    }

    std::vector<double> shiftedValues(x.begin(), x.end());
    auto baseValue = options.Scheme == DifferentiationScheme::Forward ? function(x) : 0.0;

    for (std::size_t index = 0; index < x.size(); ++index) {
        auto originalValue = shiftedValues[index];
        auto step = options.Step > 0.0 ? options.Step : DefaultStep(originalValue, options.Scheme);

        switch (options.Scheme) {
        case DifferentiationScheme::Forward:
            shiftedValues[index] = originalValue + step;
            gradient[index] =
                (function(std::span<const double>(shiftedValues.data(), shiftedValues.size())) -
                 baseValue) /
                step;
            break;
        case DifferentiationScheme::Central: {
            shiftedValues[index] = originalValue + step;
            auto plusValue =
                function(std::span<const double>(shiftedValues.data(), shiftedValues.size()));
            shiftedValues[index] = originalValue - step;
            auto minusValue =
                function(std::span<const double>(shiftedValues.data(), shiftedValues.size()));
            gradient[index] = (plusValue - minusValue) / (2.0 * step);
            break;
        }
        case DifferentiationScheme::FivePoint: {
            shiftedValues[index] = originalValue + 2.0 * step;
            auto plusTwoValue =
                function(std::span<const double>(shiftedValues.data(), shiftedValues.size()));
            shiftedValues[index] = originalValue + step;
            auto plusValue =
                function(std::span<const double>(shiftedValues.data(), shiftedValues.size()));
            shiftedValues[index] = originalValue - step;
            auto minusValue =
                function(std::span<const double>(shiftedValues.data(), shiftedValues.size()));
            shiftedValues[index] = originalValue - 2.0 * step;
            auto minusTwoValue =
                function(std::span<const double>(shiftedValues.data(), shiftedValues.size()));
            gradient[index] =
                (-plusTwoValue + 8.0 * plusValue - 8.0 * minusValue + minusTwoValue) /
                (12.0 * step);
            break;
        }
        }
        shiftedValues[index] = originalValue;
    }
}

[[nodiscard]] Vector ComputeGradient(const ObjectiveFunction& function,
                                     std::span<const double> x,
                                     const DifferentiationOptions& options)
{
    Vector gradient(x.size());
    ComputeGradient(function, x, gradient.Span(), options);
    return gradient;
}

} // namespace optlib
