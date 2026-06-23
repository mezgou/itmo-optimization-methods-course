#include <gtest/gtest.h>

#include <optlib/core/Differentiation.hpp>
#include <optlib/core/Dual.hpp>
#include <optlib/core/LinAlg.hpp>
#include <optlib/core/functions/Rosenbrock.hpp>

#include <cmath>
#include <span>
#include <vector>

namespace {

constexpr double TOLERANCE = 1e-8;

} // namespace

TEST(LinAlgTest, ComputesVectorOperations)
{
    optlib::Vector left{1.0, 2.0, 3.0};
    optlib::Vector right{4.0, -5.0, 6.0};

    EXPECT_DOUBLE_EQ(optlib::Dot(left, right), 12.0);
    EXPECT_DOUBLE_EQ(optlib::Norm2(left), std::sqrt(14.0));
    EXPECT_DOUBLE_EQ(optlib::NormInf(right), 6.0);

    optlib::Axpy(2.0, left.Span(), right.Span());
    EXPECT_DOUBLE_EQ(right[0], 6.0);
    EXPECT_DOUBLE_EQ(right[1], -1.0);
    EXPECT_DOUBLE_EQ(right[2], 12.0);
}

TEST(LinAlgTest, ComputesMatrixOperations)
{
    std::vector<double> matrixValues{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    optlib::Matrix matrix(2, 3, matrixValues);
    optlib::Vector vector{1.0, 0.5, -1.0};

    auto gemv = optlib::Gemv(matrix, vector);
    EXPECT_DOUBLE_EQ(gemv[0], -1.0);
    EXPECT_DOUBLE_EQ(gemv[1], 0.5);

    std::vector<double> rightValues{1.0, 2.0, 3.0, 4.0, -1.0, 0.5};
    optlib::Matrix right(3, 2, rightValues);
    auto gemm = optlib::Gemm(matrix, right);
    EXPECT_EQ(gemm.Rows(), 2U);
    EXPECT_EQ(gemm.Cols(), 2U);
    EXPECT_DOUBLE_EQ(gemm.At(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(gemm.At(0, 1), 11.5);
    EXPECT_DOUBLE_EQ(gemm.At(1, 0), 13.0);
    EXPECT_DOUBLE_EQ(gemm.At(1, 1), 31.0);
}

TEST(DualTest, ComputesMultidualGradient)
{
    auto x = optlib::Dual::Variable(2.0, 0, 2);
    auto y = optlib::Dual::Variable(3.0, 1, 2);

    auto result = x * x * y + optlib::Sin(y);

    EXPECT_NEAR(result.Value(), 12.0 + std::sin(3.0), TOLERANCE);
    EXPECT_NEAR(result.Gradient()[0], 12.0, TOLERANCE);
    EXPECT_NEAR(result.Gradient()[1], 4.0 + std::cos(3.0), TOLERANCE);
}

TEST(RosenbrockTest, ComputesAnalyticValueGradientAndHessian)
{
    optlib::Vector point{-1.2, 1.0};
    optlib::Rosenbrock function(point.Size());

    auto value = function.Value(point.Span());
    auto gradient = function.Gradient(point.Span());
    auto hessian = function.Hessian(point.Span());

    EXPECT_NEAR(value, 24.2, TOLERANCE);
    EXPECT_NEAR(gradient[0], -215.6, TOLERANCE);
    EXPECT_NEAR(gradient[1], -88.0, TOLERANCE);
    EXPECT_NEAR(hessian.At(0, 0), 1330.0, TOLERANCE);
    EXPECT_NEAR(hessian.At(0, 1), 480.0, TOLERANCE);
    EXPECT_NEAR(hessian.At(1, 0), 480.0, TOLERANCE);
    EXPECT_NEAR(hessian.At(1, 1), 200.0, TOLERANCE);
}

TEST(DifferentiationTest, FiniteDifferenceGradientsMatchRosenbrockGradient)
{
    optlib::Vector point{-1.2, 1.0, 0.7, 1.3};
    auto analytic = optlib::RosenbrockGradient(point.Span());
    auto function = [](std::span<const double> values) {
        return optlib::RosenbrockValue(values);
    };

    for (auto scheme : {optlib::DifferentiationScheme::Forward,
                        optlib::DifferentiationScheme::Central,
                        optlib::DifferentiationScheme::FivePoint}) {
        auto numeric = optlib::ComputeGradient(
            function, point.Span(), optlib::DifferentiationOptions{.Scheme = scheme});
        auto tolerance = scheme == optlib::DifferentiationScheme::Forward ? 5e-4 : 1e-5;
        for (std::size_t index = 0; index < point.Size(); ++index) {
            EXPECT_NEAR(numeric[index], analytic[index], tolerance);
        }
    }
}

TEST(DifferentiationTest, AutogradGradientMatchesRosenbrockGradient)
{
    optlib::Vector point{-1.2, 1.0, 0.7, 1.3, -0.4};
    auto analytic = optlib::RosenbrockGradient(point.Span());
    auto function = [](std::span<const optlib::Dual> values) {
        return optlib::Rosenbrock(values.size()).Eval<optlib::Dual>(values);
    };

    auto automatic = optlib::ComputeAutogradGradient(function, point.Span());
    for (std::size_t index = 0; index < point.Size(); ++index) {
        EXPECT_NEAR(automatic[index], analytic[index], TOLERANCE);
    }
}
