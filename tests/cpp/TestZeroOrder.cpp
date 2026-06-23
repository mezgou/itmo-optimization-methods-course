#include <gtest/gtest.h>

#include <optlib/core/optimizers/ZeroOrder.hpp>

namespace {

double QuadraticValue(std::span<const double> values)
{
    auto sum = 0.0;
    for (double value : values) {
        auto shifted = value - 1.5;
        sum += shifted * shifted;
    }
    return sum;
}

optlib::ZeroOrderConfig ConfigFor()
{
    optlib::ZeroOrderConfig config;
    config.Criteria.MaxIterations = 300;
    config.Criteria.StepTolerance = 1e-7;
    config.Criteria.FunctionTolerance = 1e-12;
    config.InitialStep = 0.8;
    config.LineSearchRadius = 2.0;
    config.StoreTrajectory = true;
    return config;
}

} // namespace

TEST(ZeroOrderTest, MethodsConvergeOnQuadratic)
{
    const optlib::Vector initialX{-2.0, 4.0};
    for (auto method : {optlib::ZeroOrderMethod::NelderMead, optlib::ZeroOrderMethod::Powell,
                        optlib::ZeroOrderMethod::CoordinateSearch}) {
        const auto result = optlib::MinimizeZeroOrder(QuadraticValue, initialX.Span(), method, ConfigFor());
        EXPECT_LT(result.Value, 1e-6);
        EXPECT_GT(result.FunctionEvaluations, 0U);
    }
}

TEST(ZeroOrderTest, MethodsReduceRosenbrock)
{
    const optlib::Vector initialX{-1.2, 1.0};
    for (auto method : {optlib::ZeroOrderMethod::NelderMead, optlib::ZeroOrderMethod::Powell,
                        optlib::ZeroOrderMethod::CoordinateSearch}) {
        auto config = ConfigFor();
        config.Criteria.MaxIterations = 600;
        const auto result = optlib::MinimizeRosenbrockZeroOrder(initialX.Span(), method, config);
        EXPECT_LT(result.Value, 1.0);
    }
}
