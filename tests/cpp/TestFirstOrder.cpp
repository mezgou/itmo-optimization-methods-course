#include <gtest/gtest.h>

#include <optlib/core/optimizers/FirstOrder.hpp>

namespace {

optlib::FirstOrderConfig ConfigFor(optlib::FirstOrderMethod method)
{
    optlib::FirstOrderConfig config;
    config.Criteria.MaxIterations = 20000;
    config.Criteria.GradientTolerance = 1e-3;
    config.Criteria.StepTolerance = 0.0;
    config.Criteria.FunctionTolerance = 0.0;
    config.StoreTrajectory = true;

    switch (method) {
    case optlib::FirstOrderMethod::GradientDescent:
        config.LearningRate = 1e-3;
        break;
    case optlib::FirstOrderMethod::HeavyBall:
        config.LearningRate = 8e-4;
        config.Momentum = 0.8;
        break;
    case optlib::FirstOrderMethod::Nesterov:
        config.LearningRate = 8e-4;
        config.Momentum = 0.8;
        break;
    case optlib::FirstOrderMethod::Adam:
        config.LearningRate = 2e-2;
        break;
    case optlib::FirstOrderMethod::RMSProp:
        config.LearningRate = 2e-4;
        config.Beta2 = 0.999;
        break;
    case optlib::FirstOrderMethod::Adagrad:
        config.LearningRate = 5e-2;
        break;
    }
    return config;
}

} // namespace

TEST(FirstOrderTest, MethodsReduceRosenbrock2D)
{
    const optlib::Vector initialX{-1.2, 1.0};
    for (const optlib::FirstOrderMethod method :
         {optlib::FirstOrderMethod::GradientDescent, optlib::FirstOrderMethod::HeavyBall,
          optlib::FirstOrderMethod::Nesterov, optlib::FirstOrderMethod::Adam,
          optlib::FirstOrderMethod::RMSProp, optlib::FirstOrderMethod::Adagrad}) {
        optlib::FirstOrderConfig config = ConfigFor(method);
        const optlib::OptimizeResult result =
            optlib::MinimizeRosenbrock(initialX.Span(), method, config);
        EXPECT_LT(result.Value, 1e-2);
        EXPECT_LT(result.GradientNorm, 0.25);
        EXPECT_GT(result.Path.Size(), 1U);
        EXPECT_EQ(result.Path.Dimension(), initialX.Size());
    }
}

TEST(FirstOrderTest, AdamConvergesOnRosenbrockND)
{
    optlib::Vector initialX{-1.2, 1.0, -1.2, 1.0, -1.2};
    optlib::FirstOrderConfig config = ConfigFor(optlib::FirstOrderMethod::Adam);
    config.Criteria.MaxIterations = 30000;
    const optlib::OptimizeResult result =
        optlib::MinimizeRosenbrock(initialX.Span(), optlib::FirstOrderMethod::Adam, config);
    EXPECT_LT(result.Value, 1e-2);
    EXPECT_LT(result.GradientNorm, 0.25);
}
