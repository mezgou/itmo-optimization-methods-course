#include <optlib/core/nn/NeuralNetwork.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace optlib {

namespace {

constexpr double EPSILON = 1e-12;

[[nodiscard]] double SigmoidValue(double value)
{
    if (value >= 0.0) {
        const auto expValue = std::exp(-value);
        return 1.0 / (1.0 + expValue);
    }
    const auto expValue = std::exp(value);
    return expValue / (1.0 + expValue);
}

[[nodiscard]] double Activate(double value, ActivationType activation)
{
    switch (activation) {
    case ActivationType::Relu:
        return std::max(0.0, value);
    case ActivationType::LeakyRelu:
        return value >= 0.0 ? value : 0.01 * value;
    case ActivationType::Sigmoid:
        return SigmoidValue(value);
    case ActivationType::Tanh:
        return std::tanh(value);
    }
    return value;
}

[[nodiscard]] double ActivationDerivativeFromPreactivation(double value, ActivationType activation)
{
    switch (activation) {
    case ActivationType::Relu:
        return value > 0.0 ? 1.0 : 0.0;
    case ActivationType::LeakyRelu:
        return value >= 0.0 ? 1.0 : 0.01;
    case ActivationType::Sigmoid: {
        const auto activated = SigmoidValue(value);
        return activated * (1.0 - activated);
    }
    case ActivationType::Tanh: {
        const auto activated = std::tanh(value);
        return 1.0 - activated * activated;
    }
    }
    return 1.0;
}

void ValidateShapes(const Matrix& features,
                    std::span<const double> targets,
                    const BinaryMlpConfig& config)
{
    if (features.Cols() != config.InputDim) {
        throw std::invalid_argument("Feature column count does not match MLP input dimension");
    }
    if (features.Rows() != targets.size()) {
        throw std::invalid_argument("Target count does not match feature row count");
    }
    if (config.InputDim == 0 || config.HiddenDim == 0) {
        throw std::invalid_argument("MLP dimensions must be positive");
    }
}

struct ParameterView {
    std::span<const double> W1;
    std::span<const double> B1;
    std::span<const double> W2;
    double B2 = 0.0;
};

struct MutableParameterView {
    std::span<double> W1;
    std::span<double> B1;
    std::span<double> W2;
    double& B2;
};

ParameterView ViewParameters(std::span<const double> parameters, const BinaryMlpConfig& config)
{
    const auto expected = BinaryMlpParameterCount(config);
    if (parameters.size() != expected) {
        throw std::invalid_argument("MLP parameter vector has unexpected size");
    }
    const auto w1Size = config.HiddenDim * config.InputDim;
    const auto b1Offset = w1Size;
    const auto w2Offset = b1Offset + config.HiddenDim;
    const auto b2Offset = w2Offset + config.HiddenDim;
    return {.W1 = parameters.subspan(0, w1Size),
            .B1 = parameters.subspan(b1Offset, config.HiddenDim),
            .W2 = parameters.subspan(w2Offset, config.HiddenDim),
            .B2 = parameters[b2Offset]};
}

MutableParameterView ViewMutableParameters(std::span<double> parameters,
                                           const BinaryMlpConfig& config)
{
    const auto expected = BinaryMlpParameterCount(config);
    if (parameters.size() != expected) {
        throw std::invalid_argument("MLP gradient vector has unexpected size");
    }
    const auto w1Size = config.HiddenDim * config.InputDim;
    const auto b1Offset = w1Size;
    const auto w2Offset = b1Offset + config.HiddenDim;
    const auto b2Offset = w2Offset + config.HiddenDim;
    return {.W1 = parameters.subspan(0, w1Size),
            .B1 = parameters.subspan(b1Offset, config.HiddenDim),
            .W2 = parameters.subspan(w2Offset, config.HiddenDim),
            .B2 = parameters[b2Offset]};
}

} // namespace

ActivationType ParseActivationType(std::string_view name)
{
    if (name == "relu" || name == "Relu") {
        return ActivationType::Relu;
    }
    if (name == "leaky_relu" || name == "LeakyRelu") {
        return ActivationType::LeakyRelu;
    }
    if (name == "sigmoid" || name == "Sigmoid") {
        return ActivationType::Sigmoid;
    }
    if (name == "tanh" || name == "Tanh") {
        return ActivationType::Tanh;
    }
    throw std::invalid_argument("Unknown activation type");
}

InitializationType ParseInitializationType(std::string_view name)
{
    if (name == "xavier" || name == "glorot" || name == "Xavier") {
        return InitializationType::Xavier;
    }
    if (name == "he" || name == "kaiming" || name == "He") {
        return InitializationType::He;
    }
    throw std::invalid_argument("Unknown initialization type");
}

std::size_t BinaryMlpParameterCount(const BinaryMlpConfig& config)
{
    return config.HiddenDim * config.InputDim + config.HiddenDim + config.HiddenDim + 1;
}

Vector InitializeBinaryMlpParameters(const BinaryMlpConfig& config,
                                     InitializationType initialization,
                                     std::size_t seed)
{
    Vector parameters(BinaryMlpParameterCount(config));
    std::mt19937_64 generator(seed);
    const auto w1Limit = initialization == InitializationType::He
        ? std::sqrt(6.0 / static_cast<double>(config.InputDim))
        : std::sqrt(6.0 / static_cast<double>(config.InputDim + config.HiddenDim));
    const auto w2Limit = initialization == InitializationType::He
        ? std::sqrt(6.0 / static_cast<double>(config.HiddenDim))
        : std::sqrt(6.0 / static_cast<double>(config.HiddenDim + 1));
    std::uniform_real_distribution<double> w1Distribution(-w1Limit, w1Limit);
    std::uniform_real_distribution<double> w2Distribution(-w2Limit, w2Limit);
    auto view = ViewMutableParameters(parameters.Span(), config);
    for (double& value : view.W1) {
        value = w1Distribution(generator);
    }
    std::fill(view.B1.begin(), view.B1.end(), 0.0);
    for (double& value : view.W2) {
        value = w2Distribution(generator);
    }
    view.B2 = 0.0;
    return parameters;
}

double BinaryCrossEntropy(std::span<const double> probabilities, std::span<const double> targets)
{
    if (probabilities.size() != targets.size()) {
        throw std::invalid_argument("Probability and target sizes do not match");
    }
    auto loss = 0.0;
    for (std::size_t index = 0; index < probabilities.size(); ++index) {
        const auto probability = std::clamp(probabilities[index], EPSILON, 1.0 - EPSILON);
        loss -= targets[index] * std::log(probability)
            + (1.0 - targets[index]) * std::log(1.0 - probability);
    }
    return loss / static_cast<double>(probabilities.size());
}

double BinaryF1Score(std::span<const double> probabilities,
                     std::span<const double> targets,
                     double threshold)
{
    if (probabilities.size() != targets.size()) {
        throw std::invalid_argument("Probability and target sizes do not match");
    }
    auto truePositive = 0.0;
    auto falsePositive = 0.0;
    auto falseNegative = 0.0;
    for (std::size_t index = 0; index < probabilities.size(); ++index) {
        const bool predicted = probabilities[index] >= threshold;
        const bool actual = targets[index] >= 0.5;
        if (predicted && actual) {
            truePositive += 1.0;
        } else if (predicted && !actual) {
            falsePositive += 1.0;
        } else if (!predicted && actual) {
            falseNegative += 1.0;
        }
    }
    const auto denominator = 2.0 * truePositive + falsePositive + falseNegative;
    return denominator > 0.0 ? 2.0 * truePositive / denominator : 0.0;
}

Vector BinaryMlpPredictProba(std::span<const double> parameters,
                             const Matrix& features,
                             const BinaryMlpConfig& config)
{
    if (features.Cols() != config.InputDim) {
        throw std::invalid_argument("Feature column count does not match MLP input dimension");
    }
    const auto view = ViewParameters(parameters, config);
    Vector probabilities(features.Rows());
    Vector hidden(config.HiddenDim);
    for (std::size_t row = 0; row < features.Rows(); ++row) {
        for (std::size_t hiddenIndex = 0; hiddenIndex < config.HiddenDim; ++hiddenIndex) {
            auto value = view.B1[hiddenIndex];
            for (std::size_t inputIndex = 0; inputIndex < config.InputDim; ++inputIndex) {
                value += view.W1[hiddenIndex * config.InputDim + inputIndex]
                    * features.At(row, inputIndex);
            }
            hidden[hiddenIndex] = Activate(value, config.Activation);
        }
        auto logit = view.B2;
        for (std::size_t hiddenIndex = 0; hiddenIndex < config.HiddenDim; ++hiddenIndex) {
            logit += view.W2[hiddenIndex] * hidden[hiddenIndex];
        }
        probabilities[row] = SigmoidValue(logit);
    }
    return probabilities;
}

double BinaryMlpLossAndGradient(std::span<const double> parameters,
                                const Matrix& features,
                                std::span<const double> targets,
                                const BinaryMlpConfig& config,
                                std::span<double> gradient)
{
    ValidateShapes(features, targets, config);
    const auto view = ViewParameters(parameters, config);
    auto gradientView = ViewMutableParameters(gradient, config);
    std::fill(gradient.begin(), gradient.end(), 0.0);

    auto loss = 0.0;
    Vector hidden(config.HiddenDim);
    Vector preactivation(config.HiddenDim);
    const auto invRows = 1.0 / static_cast<double>(features.Rows());

    for (std::size_t row = 0; row < features.Rows(); ++row) {
        for (std::size_t hiddenIndex = 0; hiddenIndex < config.HiddenDim; ++hiddenIndex) {
            auto value = view.B1[hiddenIndex];
            for (std::size_t inputIndex = 0; inputIndex < config.InputDim; ++inputIndex) {
                value += view.W1[hiddenIndex * config.InputDim + inputIndex]
                    * features.At(row, inputIndex);
            }
            preactivation[hiddenIndex] = value;
            hidden[hiddenIndex] = Activate(value, config.Activation);
        }
        auto logit = view.B2;
        for (std::size_t hiddenIndex = 0; hiddenIndex < config.HiddenDim; ++hiddenIndex) {
            logit += view.W2[hiddenIndex] * hidden[hiddenIndex];
        }
        const auto probability = SigmoidValue(logit);
        const auto clipped = std::clamp(probability, EPSILON, 1.0 - EPSILON);
        loss -= targets[row] * std::log(clipped) + (1.0 - targets[row]) * std::log(1.0 - clipped);

        const auto outputDelta = (probability - targets[row]) * invRows;
        for (std::size_t hiddenIndex = 0; hiddenIndex < config.HiddenDim; ++hiddenIndex) {
            gradientView.W2[hiddenIndex] += outputDelta * hidden[hiddenIndex];
            const auto hiddenDelta = outputDelta * view.W2[hiddenIndex]
                * ActivationDerivativeFromPreactivation(preactivation[hiddenIndex], config.Activation);
            gradientView.B1[hiddenIndex] += hiddenDelta;
            for (std::size_t inputIndex = 0; inputIndex < config.InputDim; ++inputIndex) {
                gradientView.W1[hiddenIndex * config.InputDim + inputIndex] +=
                    hiddenDelta * features.At(row, inputIndex);
            }
        }
        gradientView.B2 += outputDelta;
    }

    loss *= invRows;
    if (config.L2 > 0.0) {
        auto penalty = 0.0;
        for (std::size_t index = 0; index < parameters.size() - 1; ++index) {
            penalty += parameters[index] * parameters[index];
            gradient[index] += config.L2 * parameters[index];
        }
        loss += 0.5 * config.L2 * penalty;
    }
    return loss;
}

BinaryMlpTrainResult TrainBinaryMlp(const Matrix& features,
                                    std::span<const double> targets,
                                    const BinaryMlpTrainConfig& config)
{
    ValidateShapes(features, targets, config.Network);
    auto initialParameters =
        InitializeBinaryMlpParameters(config.Network, config.Initialization, config.Seed);
    auto valueFunction = [&features, targets, &config](std::span<const double> parameters) {
        Vector gradient(BinaryMlpParameterCount(config.Network));
        return BinaryMlpLossAndGradient(
            parameters, features, targets, config.Network, gradient.Span());
    };
    auto gradientFunction = [&features, targets, &config](std::span<const double> parameters,
                                                          std::span<double> gradient) {
        static_cast<void>(
            BinaryMlpLossAndGradient(parameters, features, targets, config.Network, gradient));
    };

    auto result = MinimizeFirstOrder(
        valueFunction, gradientFunction, initialParameters.Span(), config.Method, config.Optimizer);
    const auto probabilities = BinaryMlpPredictProba(result.X.Span(), features, config.Network);
    return {.Parameters = result.X,
            .Loss = result.Value,
            .F1 = BinaryF1Score(probabilities.Span(), targets),
            .OptimizerResult = std::move(result)};
}

} // namespace optlib
