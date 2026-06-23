#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

#include <optlib/core/LinAlg.hpp>
#include <optlib/core/OptimizeResult.hpp>
#include <optlib/core/optimizers/FirstOrder.hpp>

namespace optlib {

enum class ActivationType {
    Relu,
    LeakyRelu,
    Sigmoid,
    Tanh,
};

enum class InitializationType {
    Xavier,
    He,
};

struct BinaryMlpConfig {
    std::size_t InputDim = 2;
    std::size_t HiddenDim = 8;
    ActivationType Activation = ActivationType::Tanh;
    double L2 = 0.0;
};

struct BinaryMlpTrainConfig {
    BinaryMlpConfig Network;
    FirstOrderMethod Method = FirstOrderMethod::Adam;
    FirstOrderConfig Optimizer;
    InitializationType Initialization = InitializationType::Xavier;
    std::size_t Seed = 42;
};

struct BinaryMlpTrainResult {
    Vector Parameters;
    double Loss = 0.0;
    double F1 = 0.0;
    OptimizeResult OptimizerResult;
};

[[nodiscard]] ActivationType ParseActivationType(std::string_view name);
[[nodiscard]] InitializationType ParseInitializationType(std::string_view name);

[[nodiscard]] std::size_t BinaryMlpParameterCount(const BinaryMlpConfig& config);

[[nodiscard]] Vector InitializeBinaryMlpParameters(const BinaryMlpConfig& config,
                                                   InitializationType initialization,
                                                   std::size_t seed);

[[nodiscard]] double BinaryCrossEntropy(std::span<const double> probabilities,
                                        std::span<const double> targets);

[[nodiscard]] double BinaryF1Score(std::span<const double> probabilities,
                                   std::span<const double> targets,
                                   double threshold = 0.5);

[[nodiscard]] Vector BinaryMlpPredictProba(std::span<const double> parameters,
                                           const Matrix& features,
                                           const BinaryMlpConfig& config);

[[nodiscard]] double BinaryMlpLossAndGradient(std::span<const double> parameters,
                                              const Matrix& features,
                                              std::span<const double> targets,
                                              const BinaryMlpConfig& config,
                                              std::span<double> gradient);

[[nodiscard]] BinaryMlpTrainResult TrainBinaryMlp(const Matrix& features,
                                                  std::span<const double> targets,
                                                  const BinaryMlpTrainConfig& config);

} // namespace optlib
