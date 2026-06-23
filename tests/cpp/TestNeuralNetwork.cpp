#include <gtest/gtest.h>

#include <optlib/core/Differentiation.hpp>
#include <optlib/core/nn/NeuralNetwork.hpp>

#include <vector>

namespace {

optlib::Matrix XorFeatures()
{
    const std::vector<double> values{0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};
    return optlib::Matrix(4, 2, values);
}

optlib::Vector XorTargets()
{
    return optlib::Vector{0.0, 1.0, 1.0, 0.0};
}

} // namespace

TEST(NeuralNetworkTest, BackpropMatchesFiniteDifference)
{
    const auto features = XorFeatures();
    const auto targets = XorTargets();
    optlib::BinaryMlpConfig config;
    config.InputDim = 2;
    config.HiddenDim = 3;
    config.Activation = optlib::ActivationType::Tanh;
    auto parameters =
        optlib::InitializeBinaryMlpParameters(config, optlib::InitializationType::Xavier, 7);
    optlib::Vector gradient(optlib::BinaryMlpParameterCount(config));
    const auto loss = optlib::BinaryMlpLossAndGradient(
        parameters.Span(), features, targets.Span(), config, gradient.Span());
    EXPECT_GT(loss, 0.0);

    auto numericGradient = optlib::ComputeGradient(
        [&features, &targets, &config](std::span<const double> values) {
            optlib::Vector scratch(optlib::BinaryMlpParameterCount(config));
            return optlib::BinaryMlpLossAndGradient(
                values, features, targets.Span(), config, scratch.Span());
        },
        parameters.Span(),
        optlib::DifferentiationOptions{optlib::DifferentiationScheme::Central, 1e-5});

    for (std::size_t index = 0; index < gradient.Size(); ++index) {
        EXPECT_NEAR(gradient[index], numericGradient[index], 1e-5);
    }
}

TEST(NeuralNetworkTest, TrainsXorWithAdam)
{
    const auto features = XorFeatures();
    const auto targets = XorTargets();
    optlib::BinaryMlpTrainConfig config;
    config.Network.InputDim = 2;
    config.Network.HiddenDim = 8;
    config.Network.Activation = optlib::ActivationType::Tanh;
    config.Method = optlib::FirstOrderMethod::Adam;
    config.Optimizer.LearningRate = 0.05;
    config.Optimizer.Criteria.MaxIterations = 5000;
    config.Optimizer.Criteria.GradientTolerance = 1e-5;
    config.Optimizer.Criteria.StepTolerance = 0.0;
    config.Optimizer.Criteria.FunctionTolerance = 0.0;
    config.Optimizer.StoreTrajectory = false;
    config.Seed = 11;

    const auto result = optlib::TrainBinaryMlp(features, targets.Span(), config);
    EXPECT_LT(result.Loss, 0.05);
    EXPECT_GT(result.F1, 0.99);
}
