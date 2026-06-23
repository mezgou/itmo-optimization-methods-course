#include <gtest/gtest.h>

#include <optlib/core/Differentiation.hpp>
#include <optlib/core/functions/BenchmarkFunctions.hpp>

namespace {

void ExpectGradientMatchesFiniteDifference(std::span<const double> point,
                                           std::span<const double> analyticGradient,
                                           const optlib::ObjectiveFunction& valueFunction,
                                           double tolerance)
{
    const auto numericGradient = optlib::ComputeGradient(
        valueFunction,
        point,
        optlib::DifferentiationOptions{optlib::DifferentiationScheme::Central, 0.0});
    ASSERT_EQ(numericGradient.Size(), analyticGradient.size());
    for (std::size_t index = 0; index < analyticGradient.size(); ++index) {
        EXPECT_NEAR(analyticGradient[index], numericGradient[index], tolerance);
    }
}

void ExpectSymmetric(const optlib::Matrix& matrix)
{
    ASSERT_EQ(matrix.Rows(), matrix.Cols());
    for (std::size_t row = 0; row < matrix.Rows(); ++row) {
        for (std::size_t col = 0; col < matrix.Cols(); ++col) {
            EXPECT_NEAR(matrix.At(row, col), matrix.At(col, row), 1e-12);
        }
    }
}

} // namespace

TEST(BenchmarkFunctionsTest, KnownMinimaHaveExpectedValues)
{
    EXPECT_NEAR(optlib::RastriginValue(optlib::Vector{0.0, 0.0}.Span()), 0.0, 1e-12);
    EXPECT_NEAR(optlib::HimmelblauValue(optlib::Vector{3.0, 2.0}.Span()), 0.0, 1e-12);
    EXPECT_NEAR(optlib::AckleyValue(optlib::Vector{0.0, 0.0, 0.0}.Span()), 0.0, 1e-12);
    EXPECT_NEAR(optlib::BealeValue(optlib::Vector{3.0, 0.5}.Span()), 0.0, 1e-12);
    EXPECT_NEAR(optlib::BoothValue(optlib::Vector{1.0, 3.0}.Span()), 0.0, 1e-12);
    EXPECT_NEAR(
        optlib::StyblinskiTangValue(optlib::Vector{-2.903534}.Span()), -39.16616570377142, 1e-5);
}

TEST(BenchmarkFunctionsTest, AnalyticGradientsMatchFiniteDifferences)
{
    const optlib::Vector rastriginPoint{0.2, -0.4, 0.7};
    const auto rastriginGradient = optlib::RastriginGradient(rastriginPoint.Span());
    ExpectGradientMatchesFiniteDifference(
        rastriginPoint.Span(),
        rastriginGradient.Span(),
        [](std::span<const double> values) { return optlib::RastriginValue(values); },
        1e-5);

    const optlib::Vector himmelblauPoint{2.5, 1.5};
    const auto himmelblauGradient = optlib::HimmelblauGradient(himmelblauPoint.Span());
    ExpectGradientMatchesFiniteDifference(
        himmelblauPoint.Span(),
        himmelblauGradient.Span(),
        [](std::span<const double> values) { return optlib::HimmelblauValue(values); },
        1e-5);

    const optlib::Vector bealePoint{2.0, 0.25};
    const auto bealeGradient = optlib::BealeGradient(bealePoint.Span());
    ExpectGradientMatchesFiniteDifference(
        bealePoint.Span(),
        bealeGradient.Span(),
        [](std::span<const double> values) { return optlib::BealeValue(values); },
        1e-5);
}

TEST(BenchmarkFunctionsTest, HessiansAreSymmetric)
{
    ExpectSymmetric(optlib::RastriginHessian(optlib::Vector{0.2, -0.4, 0.7}.Span()));
    ExpectSymmetric(optlib::HimmelblauHessian(optlib::Vector{2.5, 1.5}.Span()));
    ExpectSymmetric(optlib::BealeHessian(optlib::Vector{2.0, 0.25}.Span()));
    ExpectSymmetric(optlib::BoothHessian(optlib::Vector{1.0, 3.0}.Span()));
}

TEST(BenchmarkFunctionsTest, DesmosSurfaceUsesExactFormulaAndScale)
{
    const optlib::Vector point{1.0, 1.0};
    EXPECT_NEAR(optlib::DesmosSurfaceValue(point.Span()), 73.0, 1e-12);
    EXPECT_NEAR(optlib::DesmosSurfaceValue(point.Span(), 2.0), 146.0, 1e-12);
}

TEST(BenchmarkFunctionsTest, RegistryRejectsDesmosGradient)
{
    const optlib::Vector point{1.0, 1.0};
    EXPECT_FALSE(optlib::GetBenchmarkFunctionInfo("desmos_surface", 2).IsSmooth);
    EXPECT_THROW(
        static_cast<void>(optlib::BenchmarkFunctionGradient("desmos_surface", point.Span())),
        std::invalid_argument);
}
