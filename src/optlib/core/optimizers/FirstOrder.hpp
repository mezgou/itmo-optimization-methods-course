#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <string_view>

#include <optlib/core/LinAlg.hpp>
#include <optlib/core/OptimizeResult.hpp>
#include <optlib/core/StopCriteria.hpp>
#include <optlib/core/schedulers/LearningRate.hpp>

namespace optlib {

enum class FirstOrderMethod {
    GradientDescent,
    HeavyBall,
    Nesterov,
    Adam,
    RMSProp,
    Adagrad,
};

struct FirstOrderConfig {
    double LearningRate = 1e-3;
    LearningRateSchedule Schedule = LearningRateSchedule::Constant;
    double LearningRateGamma = 0.5;
    std::size_t LearningRateStepSize = 100;
    double LearningRateDecay = 1e-3;
    double MinimumLearningRate = 0.0;
    std::size_t WarmupSteps = 0;
    std::size_t ScheduleIterations = 0;
    double Momentum = 0.9;
    double Beta1 = 0.9;
    double Beta2 = 0.999;
    double Epsilon = 1e-8;
    bool StoreTrajectory = true;
    StopCriteria Criteria;
};

using ValueFunction = std::function<double(std::span<const double>)>;
using GradientFunction = std::function<void(std::span<const double>, std::span<double>)>;

[[nodiscard]] FirstOrderMethod ParseFirstOrderMethod(std::string_view methodName);

[[nodiscard]] OptimizeResult MinimizeFirstOrder(const ValueFunction& valueFunction,
                                                const GradientFunction& gradientFunction,
                                                std::span<const double> x0,
                                                FirstOrderMethod method,
                                                const FirstOrderConfig& config = {});

[[nodiscard]] OptimizeResult MinimizeRosenbrock(std::span<const double> x0,
                                                FirstOrderMethod method,
                                                const FirstOrderConfig& config = {});

} // namespace optlib
