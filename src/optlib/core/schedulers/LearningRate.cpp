#include <optlib/core/schedulers/LearningRate.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace optlib {

LearningRateSchedule ParseLearningRateSchedule(std::string_view scheduleName)
{
    if (scheduleName == "constant" || scheduleName == "Constant") {
        return LearningRateSchedule::Constant;
    }
    if (scheduleName == "step" || scheduleName == "Step") {
        return LearningRateSchedule::Step;
    }
    if (scheduleName == "exponential" || scheduleName == "exp" || scheduleName == "Exponential") {
        return LearningRateSchedule::Exponential;
    }
    if (scheduleName == "cosine" || scheduleName == "cosine_annealing"
        || scheduleName == "Cosine") {
        return LearningRateSchedule::Cosine;
    }
    throw std::invalid_argument("Unknown learning-rate schedule");
}

double LearningRateAt(std::size_t iteration, const LearningRateConfig& config)
{
    if (config.InitialLearningRate <= 0.0) {
        throw std::invalid_argument("Initial learning rate must be positive");
    }

    auto learningRate = config.InitialLearningRate;
    switch (config.Schedule) {
    case LearningRateSchedule::Constant:
        break;
    case LearningRateSchedule::Step: {
        const auto stepSize = std::max<std::size_t>(1, config.StepSize);
        const auto stepIndex = iteration / stepSize;
        learningRate *= std::pow(config.Gamma, static_cast<double>(stepIndex));
        break;
    }
    case LearningRateSchedule::Exponential:
        learningRate *= std::exp(-config.DecayRate * static_cast<double>(iteration));
        break;
    case LearningRateSchedule::Cosine: {
        const auto totalIterations = std::max<std::size_t>(1, config.TotalIterations);
        const auto clippedIteration = std::min(iteration, totalIterations);
        const auto progress =
            static_cast<double>(clippedIteration) / static_cast<double>(totalIterations);
        learningRate = config.MinimumLearningRate
            + 0.5 * (config.InitialLearningRate - config.MinimumLearningRate)
                * (1.0 + std::cos(std::numbers::pi * progress));
        break;
    }
    }

    if (config.WarmupSteps > 0 && iteration < config.WarmupSteps) {
        const auto warmupScale =
            static_cast<double>(iteration + 1) / static_cast<double>(config.WarmupSteps);
        learningRate *= warmupScale;
    }

    return std::max(config.MinimumLearningRate, learningRate);
}

} // namespace optlib
