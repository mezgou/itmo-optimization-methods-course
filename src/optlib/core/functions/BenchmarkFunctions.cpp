#include <optlib/core/functions/BenchmarkFunctions.hpp>

#include <optlib/core/Differentiation.hpp>
#include <optlib/core/Dual.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace optlib {

namespace {

constexpr double TWO_PI = 2.0 * std::numbers::pi;

void RequireDimension(std::span<const double> x, std::size_t dimension, const char* name)
{
    if (x.size() != dimension) {
        throw std::invalid_argument(std::string(name) + " expects a fixed input dimension");
    }
}

void RequireNonEmpty(std::span<const double> x, const char* name)
{
    if (x.empty()) {
        throw std::invalid_argument(std::string(name) + " expects a non-empty input");
    }
}

Matrix DiagonalMatrix(std::span<const double> diagonal)
{
    Matrix matrix(diagonal.size(), diagonal.size());
    matrix.Fill(0.0);
    for (std::size_t index = 0; index < diagonal.size(); ++index) {
        matrix.At(index, index) = diagonal[index];
    }
    return matrix;
}

} // namespace

double RastriginValue(std::span<const double> x)
{
    RequireNonEmpty(x, "Rastrigin");
    auto value = 10.0 * static_cast<double>(x.size());
    for (double coordinate : x) {
        value += coordinate * coordinate - 10.0 * std::cos(TWO_PI * coordinate);
    }
    return value;
}

Vector RastriginGradient(std::span<const double> x)
{
    RequireNonEmpty(x, "Rastrigin");
    Vector gradient(x.size());
    for (std::size_t index = 0; index < x.size(); ++index) {
        gradient[index] = 2.0 * x[index] + 10.0 * TWO_PI * std::sin(TWO_PI * x[index]);
    }
    return gradient;
}

Matrix RastriginHessian(std::span<const double> x)
{
    RequireNonEmpty(x, "Rastrigin");
    Vector diagonal(x.size());
    for (std::size_t index = 0; index < x.size(); ++index) {
        diagonal[index] = 2.0 + 10.0 * TWO_PI * TWO_PI * std::cos(TWO_PI * x[index]);
    }
    return DiagonalMatrix(diagonal.Span());
}

Vector RastriginAutogradGradient(std::span<const double> x)
{
    return ComputeAutogradGradient(
        [](std::span<const Dual> values) {
            auto result = Dual{10.0 * static_cast<double>(values.size())};
            for (const auto& coordinate : values) {
                result = result + coordinate * coordinate - 10.0 * Cos(TWO_PI * coordinate);
            }
            return result;
        },
        x);
}

double HimmelblauValue(std::span<const double> x)
{
    RequireDimension(x, 2, "Himmelblau");
    const auto first = x[0] * x[0] + x[1] - 11.0;
    const auto second = x[0] + x[1] * x[1] - 7.0;
    return first * first + second * second;
}

Vector HimmelblauGradient(std::span<const double> x)
{
    RequireDimension(x, 2, "Himmelblau");
    const auto first = x[0] * x[0] + x[1] - 11.0;
    const auto second = x[0] + x[1] * x[1] - 7.0;
    return Vector{4.0 * x[0] * first + 2.0 * second,
                  2.0 * first + 4.0 * x[1] * second};
}

Matrix HimmelblauHessian(std::span<const double> x)
{
    RequireDimension(x, 2, "Himmelblau");
    Matrix hessian(2, 2);
    hessian.At(0, 0) = 12.0 * x[0] * x[0] + 4.0 * x[1] - 42.0;
    hessian.At(0, 1) = 4.0 * (x[0] + x[1]);
    hessian.At(1, 0) = hessian.At(0, 1);
    hessian.At(1, 1) = 4.0 * x[0] + 12.0 * x[1] * x[1] - 26.0;
    return hessian;
}

Vector HimmelblauAutogradGradient(std::span<const double> x)
{
    RequireDimension(x, 2, "Himmelblau");
    return ComputeAutogradGradient(
        [](std::span<const Dual> values) {
            const auto first = values[0] * values[0] + values[1] - 11.0;
            const auto second = values[0] + values[1] * values[1] - 7.0;
            return first * first + second * second;
        },
        x);
}

double AckleyValue(std::span<const double> x)
{
    RequireNonEmpty(x, "Ackley");
    auto sumSquares = 0.0;
    auto sumCos = 0.0;
    for (double coordinate : x) {
        sumSquares += coordinate * coordinate;
        sumCos += std::cos(TWO_PI * coordinate);
    }
    const auto dimension = static_cast<double>(x.size());
    return -20.0 * std::exp(-0.2 * std::sqrt(sumSquares / dimension))
        - std::exp(sumCos / dimension) + 20.0 + std::numbers::e;
}

Vector AckleyGradient(std::span<const double> x)
{
    RequireNonEmpty(x, "Ackley");
    const auto dimension = static_cast<double>(x.size());
    auto sumSquares = 0.0;
    auto sumCos = 0.0;
    for (double coordinate : x) {
        sumSquares += coordinate * coordinate;
        sumCos += std::cos(TWO_PI * coordinate);
    }

    Vector gradient(x.size());
    const auto root = std::sqrt(sumSquares / dimension);
    const auto expSquares = std::exp(-0.2 * root);
    const auto expCos = std::exp(sumCos / dimension);
    for (std::size_t index = 0; index < x.size(); ++index) {
        const auto squareTerm =
            root > 0.0 ? 4.0 * expSquares * x[index] / (dimension * root) : 0.0;
        const auto cosTerm = expCos * TWO_PI * std::sin(TWO_PI * x[index]) / dimension;
        gradient[index] = squareTerm + cosTerm;
    }
    return gradient;
}

Vector AckleyAutogradGradient(std::span<const double> x)
{
    RequireNonEmpty(x, "Ackley");
    return ComputeAutogradGradient(
        [](std::span<const Dual> values) {
            auto sumSquares = Dual{0.0};
            auto sumCos = Dual{0.0};
            for (const auto& coordinate : values) {
                sumSquares = sumSquares + coordinate * coordinate;
                sumCos = sumCos + Cos(TWO_PI * coordinate);
            }
            const auto dimension = static_cast<double>(values.size());
            return -20.0 * Exp(-0.2 * Sqrt(sumSquares / dimension))
                - Exp(sumCos / dimension) + 20.0 + std::numbers::e;
        },
        x);
}

double BealeValue(std::span<const double> x)
{
    RequireDimension(x, 2, "Beale");
    const auto first = 1.5 - x[0] + x[0] * x[1];
    const auto second = 2.25 - x[0] + x[0] * x[1] * x[1];
    const auto third = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
    return first * first + second * second + third * third;
}

Vector BealeGradient(std::span<const double> x)
{
    RequireDimension(x, 2, "Beale");
    const auto y = x[1];
    const double residuals[] = {1.5 - x[0] + x[0] * y,
                                2.25 - x[0] + x[0] * y * y,
                                2.625 - x[0] + x[0] * y * y * y};
    const double dx[] = {y - 1.0, y * y - 1.0, y * y * y - 1.0};
    const double dy[] = {x[0], 2.0 * x[0] * y, 3.0 * x[0] * y * y};

    auto gradX = 0.0;
    auto gradY = 0.0;
    for (std::size_t index = 0; index < 3; ++index) {
        gradX += 2.0 * residuals[index] * dx[index];
        gradY += 2.0 * residuals[index] * dy[index];
    }
    return Vector{gradX, gradY};
}

Matrix BealeHessian(std::span<const double> x)
{
    RequireDimension(x, 2, "Beale");
    const auto y = x[1];
    const double residuals[] = {1.5 - x[0] + x[0] * y,
                                2.25 - x[0] + x[0] * y * y,
                                2.625 - x[0] + x[0] * y * y * y};
    const double dx[] = {y - 1.0, y * y - 1.0, y * y * y - 1.0};
    const double dy[] = {x[0], 2.0 * x[0] * y, 3.0 * x[0] * y * y};
    const double dxy[] = {1.0, 2.0 * y, 3.0 * y * y};
    const double dyy[] = {0.0, 2.0 * x[0], 6.0 * x[0] * y};

    auto hxx = 0.0;
    auto hxy = 0.0;
    auto hyy = 0.0;
    for (std::size_t index = 0; index < 3; ++index) {
        hxx += 2.0 * dx[index] * dx[index];
        hxy += 2.0 * (dx[index] * dy[index] + residuals[index] * dxy[index]);
        hyy += 2.0 * (dy[index] * dy[index] + residuals[index] * dyy[index]);
    }
    Matrix hessian(2, 2);
    hessian.At(0, 0) = hxx;
    hessian.At(0, 1) = hxy;
    hessian.At(1, 0) = hxy;
    hessian.At(1, 1) = hyy;
    return hessian;
}

double BoothValue(std::span<const double> x)
{
    RequireDimension(x, 2, "Booth");
    const auto first = x[0] + 2.0 * x[1] - 7.0;
    const auto second = 2.0 * x[0] + x[1] - 5.0;
    return first * first + second * second;
}

Vector BoothGradient(std::span<const double> x)
{
    RequireDimension(x, 2, "Booth");
    const auto first = x[0] + 2.0 * x[1] - 7.0;
    const auto second = 2.0 * x[0] + x[1] - 5.0;
    return Vector{2.0 * first + 4.0 * second, 4.0 * first + 2.0 * second};
}

Matrix BoothHessian(std::span<const double> x)
{
    RequireDimension(x, 2, "Booth");
    Matrix hessian(2, 2);
    hessian.At(0, 0) = 10.0;
    hessian.At(0, 1) = 8.0;
    hessian.At(1, 0) = 8.0;
    hessian.At(1, 1) = 10.0;
    return hessian;
}

double StyblinskiTangValue(std::span<const double> x)
{
    RequireNonEmpty(x, "StyblinskiTang");
    auto value = 0.0;
    for (double coordinate : x) {
        value += coordinate * coordinate * coordinate * coordinate
            - 16.0 * coordinate * coordinate + 5.0 * coordinate;
    }
    return 0.5 * value;
}

Vector StyblinskiTangGradient(std::span<const double> x)
{
    RequireNonEmpty(x, "StyblinskiTang");
    Vector gradient(x.size());
    for (std::size_t index = 0; index < x.size(); ++index) {
        gradient[index] = 2.0 * x[index] * x[index] * x[index] - 16.0 * x[index] + 2.5;
    }
    return gradient;
}

Matrix StyblinskiTangHessian(std::span<const double> x)
{
    RequireNonEmpty(x, "StyblinskiTang");
    Vector diagonal(x.size());
    for (std::size_t index = 0; index < x.size(); ++index) {
        diagonal[index] = 6.0 * x[index] * x[index] - 16.0;
    }
    return DiagonalMatrix(diagonal.Span());
}

double DesmosSurfaceValue(std::span<const double> x, double scale)
{
    RequireDimension(x, 2, "DesmosSurface");
    if (scale <= 0.0) {
        throw std::invalid_argument("DesmosSurface scale must be positive");
    }
    const auto xValue = x[0];
    const auto yValue = x[1];
    const auto yBand = std::round(std::sin(10.0 * yValue)) + 2.0;
    const auto xBand = std::round(std::sin(7.0 * xValue)) + 2.0;
    const auto first = (xValue * yBand) * (xValue * yBand) + yValue - 10.0;
    const auto second = xValue + (yValue * xBand) * (yValue * xBand) - 7.0;
    return scale * (first * first + second * second);
}

Vector DesmosSurfaceNumericalGradient(std::span<const double> x,
                                      double scale,
                                      DifferentiationScheme scheme)
{
    return ComputeGradient(
        [scale](std::span<const double> values) { return DesmosSurfaceValue(values, scale); },
        x,
        DifferentiationOptions{scheme, 0.0});
}

BenchmarkFunctionInfo GetBenchmarkFunctionInfo(std::string_view name, std::size_t dimension)
{
    if (name == "rastrigin") {
        return {.Name = "rastrigin",
                .FixedDimension = 0,
                .HasGradient = true,
                .HasHessian = true,
                .IsSmooth = true,
                .IsMultimodal = true,
                .KnownMinimum = Vector(dimension),
                .KnownMinimumValue = 0.0,
                .SuggestedLowerBound = -5.12,
                .SuggestedUpperBound = 5.12};
    }
    if (name == "ackley") {
        return {.Name = "ackley",
                .FixedDimension = 0,
                .HasGradient = true,
                .HasHessian = false,
                .IsSmooth = true,
                .IsMultimodal = true,
                .KnownMinimum = Vector(dimension),
                .KnownMinimumValue = 0.0,
                .SuggestedLowerBound = -5.0,
                .SuggestedUpperBound = 5.0};
    }
    if (name == "styblinski_tang" || name == "styblinski-tang") {
        Vector minimum(dimension);
        minimum.Fill(-2.903534);
        return {.Name = "styblinski_tang",
                .FixedDimension = 0,
                .HasGradient = true,
                .HasHessian = true,
                .IsSmooth = true,
                .IsMultimodal = true,
                .KnownMinimum = std::move(minimum),
                .KnownMinimumValue = -39.16616570377142 * static_cast<double>(dimension),
                .SuggestedLowerBound = -5.0,
                .SuggestedUpperBound = 5.0};
    }
    if (name == "himmelblau") {
        return {.Name = "himmelblau",
                .FixedDimension = 2,
                .HasGradient = true,
                .HasHessian = true,
                .IsSmooth = true,
                .IsMultimodal = true,
                .KnownMinimum = Vector{3.0, 2.0},
                .KnownMinimumValue = 0.0,
                .SuggestedLowerBound = -6.0,
                .SuggestedUpperBound = 6.0};
    }
    if (name == "beale") {
        return {.Name = "beale",
                .FixedDimension = 2,
                .HasGradient = true,
                .HasHessian = true,
                .IsSmooth = true,
                .IsMultimodal = false,
                .KnownMinimum = Vector{3.0, 0.5},
                .KnownMinimumValue = 0.0,
                .SuggestedLowerBound = -4.5,
                .SuggestedUpperBound = 4.5};
    }
    if (name == "booth") {
        return {.Name = "booth",
                .FixedDimension = 2,
                .HasGradient = true,
                .HasHessian = true,
                .IsSmooth = true,
                .IsMultimodal = false,
                .KnownMinimum = Vector{1.0, 3.0},
                .KnownMinimumValue = 0.0,
                .SuggestedLowerBound = -10.0,
                .SuggestedUpperBound = 10.0};
    }
    if (name == "desmos_surface" || name == "desmos") {
        return {.Name = "desmos_surface",
                .FixedDimension = 2,
                .HasGradient = false,
                .HasHessian = false,
                .IsSmooth = false,
                .IsMultimodal = true,
                .KnownMinimum = Vector(),
                .KnownMinimumValue = 0.0,
                .SuggestedLowerBound = -6.0,
                .SuggestedUpperBound = 6.0};
    }
    throw std::invalid_argument("Unknown benchmark function");
}

double BenchmarkFunctionValue(std::string_view name, std::span<const double> x, double scale)
{
    if (name == "rastrigin") {
        return RastriginValue(x);
    }
    if (name == "ackley") {
        return AckleyValue(x);
    }
    if (name == "styblinski_tang" || name == "styblinski-tang") {
        return StyblinskiTangValue(x);
    }
    if (name == "himmelblau") {
        return HimmelblauValue(x);
    }
    if (name == "beale") {
        return BealeValue(x);
    }
    if (name == "booth") {
        return BoothValue(x);
    }
    if (name == "desmos_surface" || name == "desmos") {
        return DesmosSurfaceValue(x, scale);
    }
    throw std::invalid_argument("Unknown benchmark function");
}

Vector BenchmarkFunctionGradient(std::string_view name, std::span<const double> x)
{
    if (name == "rastrigin") {
        return RastriginGradient(x);
    }
    if (name == "ackley") {
        return AckleyGradient(x);
    }
    if (name == "styblinski_tang" || name == "styblinski-tang") {
        return StyblinskiTangGradient(x);
    }
    if (name == "himmelblau") {
        return HimmelblauGradient(x);
    }
    if (name == "beale") {
        return BealeGradient(x);
    }
    if (name == "booth") {
        return BoothGradient(x);
    }
    if (name == "desmos_surface" || name == "desmos") {
        throw std::invalid_argument("DesmosSurface is discontinuous; gradient is not available");
    }
    throw std::invalid_argument("Benchmark function gradient is not available");
}

Matrix BenchmarkFunctionHessian(std::string_view name, std::span<const double> x)
{
    if (name == "rastrigin") {
        return RastriginHessian(x);
    }
    if (name == "styblinski_tang" || name == "styblinski-tang") {
        return StyblinskiTangHessian(x);
    }
    if (name == "himmelblau") {
        return HimmelblauHessian(x);
    }
    if (name == "beale") {
        return BealeHessian(x);
    }
    if (name == "booth") {
        return BoothHessian(x);
    }
    if (name == "desmos_surface" || name == "desmos") {
        throw std::invalid_argument("DesmosSurface is discontinuous; Hessian is not available");
    }
    throw std::invalid_argument("Benchmark function Hessian is not available");
}

} // namespace optlib
