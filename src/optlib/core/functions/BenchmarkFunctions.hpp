#pragma once

#include <span>
#include <string_view>

#include <optlib/core/LinAlg.hpp>

namespace optlib {

enum class DifferentiationScheme;

[[nodiscard]] double RastriginValue(std::span<const double> x);
[[nodiscard]] Vector RastriginGradient(std::span<const double> x);
[[nodiscard]] Matrix RastriginHessian(std::span<const double> x);
[[nodiscard]] Vector RastriginAutogradGradient(std::span<const double> x);

[[nodiscard]] double HimmelblauValue(std::span<const double> x);
[[nodiscard]] Vector HimmelblauGradient(std::span<const double> x);
[[nodiscard]] Matrix HimmelblauHessian(std::span<const double> x);
[[nodiscard]] Vector HimmelblauAutogradGradient(std::span<const double> x);

[[nodiscard]] double AckleyValue(std::span<const double> x);
[[nodiscard]] Vector AckleyGradient(std::span<const double> x);
[[nodiscard]] Vector AckleyAutogradGradient(std::span<const double> x);

[[nodiscard]] double BealeValue(std::span<const double> x);
[[nodiscard]] Vector BealeGradient(std::span<const double> x);
[[nodiscard]] Matrix BealeHessian(std::span<const double> x);

[[nodiscard]] double BoothValue(std::span<const double> x);
[[nodiscard]] Vector BoothGradient(std::span<const double> x);
[[nodiscard]] Matrix BoothHessian(std::span<const double> x);

[[nodiscard]] double StyblinskiTangValue(std::span<const double> x);
[[nodiscard]] Vector StyblinskiTangGradient(std::span<const double> x);
[[nodiscard]] Matrix StyblinskiTangHessian(std::span<const double> x);

[[nodiscard]] double DesmosSurfaceValue(std::span<const double> x, double scale = 1.0);
[[nodiscard]] Vector DesmosSurfaceNumericalGradient(std::span<const double> x,
                                                    double scale,
                                                    DifferentiationScheme scheme);

struct BenchmarkFunctionInfo {
    std::string_view Name;
    std::size_t FixedDimension = 0;
    bool HasGradient = false;
    bool HasHessian = false;
    bool IsSmooth = true;
    bool IsMultimodal = false;
    Vector KnownMinimum;
    double KnownMinimumValue = 0.0;
    double SuggestedLowerBound = -5.0;
    double SuggestedUpperBound = 5.0;
};

[[nodiscard]] BenchmarkFunctionInfo GetBenchmarkFunctionInfo(std::string_view name,
                                                             std::size_t dimension = 2);

[[nodiscard]] double BenchmarkFunctionValue(std::string_view name,
                                            std::span<const double> x,
                                            double scale = 1.0);

[[nodiscard]] Vector BenchmarkFunctionGradient(std::string_view name, std::span<const double> x);

[[nodiscard]] Matrix BenchmarkFunctionHessian(std::string_view name, std::span<const double> x);

} // namespace optlib
