#pragma once

#include <cstddef>
#include <string_view>

namespace optlib {

enum class LearningRateSchedule {
    Constant,
    Step,
    Exponential,
    Cosine,
};

struct LearningRateConfig {
    LearningRateSchedule Schedule = LearningRateSchedule::Constant;
    double InitialLearningRate = 1e-3;
    double Gamma = 0.5;
    std::size_t StepSize = 100;
    double DecayRate = 1e-3;
    double MinimumLearningRate = 0.0;
    std::size_t TotalIterations = 1000;
    std::size_t WarmupSteps = 0;
};

[[nodiscard]] LearningRateSchedule ParseLearningRateSchedule(std::string_view scheduleName);

[[nodiscard]] double LearningRateAt(std::size_t iteration, const LearningRateConfig& config);

} // namespace optlib
