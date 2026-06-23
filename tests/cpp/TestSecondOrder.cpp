#include <gtest/gtest.h>

#include <optlib/core/functions/Rosenbrock.hpp>
#include <optlib/core/optimizers/SecondOrder.hpp>

namespace {

double QuadraticValue(std::span<const double> values)
{
    auto sum = 0.0;
    for (double value : values) {
        auto shifted = value - 2.0;
        sum += 0.5 * shifted * shifted;
    }
    return sum;
}

void QuadraticGradient(std::span<const double> values, std::span<double> gradient)
{
    for (std::size_t index = 0; index < values.size(); ++index) {
        gradient[index] = values[index] - 2.0;
    }
}

void QuadraticHessian(std::span<const double> values, optlib::Matrix& hessian)
{
    hessian.Resize(values.size(), values.size());
    hessian.Fill(0.0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        hessian.At(index, index) = 1.0;
    }
}

optlib::SecondOrderConfig FastConfig(std::size_t maxIterations = 200)
{
    optlib::SecondOrderConfig config;
    config.Criteria.MaxIterations = maxIterations;
    config.Criteria.GradientTolerance = 1e-6;
    config.Criteria.StepTolerance = 0.0;
    config.Criteria.FunctionTolerance = 0.0;
    config.StoreTrajectory = true;
    return config;
}

} // namespace

TEST(SecondOrderTest, NewtonSolvesQuadratic)
{
    const optlib::Vector initialX{-4.0, 7.0, 0.5};
    const auto result = optlib::MinimizeSecondOrder(QuadraticValue,
                                                   QuadraticGradient,
                                                   QuadraticHessian,
                                                   initialX.Span(),
                                                   optlib::SecondOrderMethod::Newton,
                                                   FastConfig());

    EXPECT_TRUE(result.Converged);
    EXPECT_LT(result.Value, 1e-12);
    EXPECT_LT(result.GradientNorm, 1e-6);
}

TEST(SecondOrderTest, BfgsAndLbfgsConvergeOnRosenbrock2D)
{
    const optlib::Vector initialX{-1.2, 1.0};
    for (auto method : {optlib::SecondOrderMethod::BFGS, optlib::SecondOrderMethod::LBFGS}) {
        auto config = FastConfig(1000);
        const auto result = optlib::MinimizeRosenbrockSecondOrder(initialX.Span(), method, config);
        EXPECT_LT(result.Value, 1e-8);
        EXPECT_LT(result.GradientNorm, 1e-4);
        EXPECT_GT(result.Path.Size(), 1U);
    }
}

TEST(SecondOrderTest, LbfgsScalesToRosenbrock100D)
{
    optlib::Vector initialX(100);
    for (std::size_t index = 0; index < initialX.Size(); ++index) {
        initialX[index] = index % 2 == 0 ? -1.2 : 1.0;
    }
    auto config = FastConfig(3000);
    config.StoreTrajectory = false;
    config.HistorySize = 12;
    const auto result =
        optlib::MinimizeRosenbrockSecondOrder(initialX.Span(), optlib::SecondOrderMethod::LBFGS, config);

    EXPECT_LT(result.Value, 1e-4);
    EXPECT_LT(result.GradientNorm, 1e-2);
}
