#include <optlib/core/optimizers/FirstOrder.hpp>

#include <optlib/core/functions/Rosenbrock.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace optlib {

namespace {

void Subtract(std::span<const double> left, std::span<const double> right, std::span<double> result)
{
    if (left.size() != right.size() || left.size() != result.size()) {
        throw std::invalid_argument("Subtract vector sizes do not match");
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        result[index] = left[index] - right[index];
    }
}

bool ShouldStop(double gradientNorm, double stepNorm, double functionDelta, StopCriteria criteria)
{
    return gradientNorm <= criteria.GradientTolerance
        || (criteria.StepTolerance > 0.0 && stepNorm <= criteria.StepTolerance)
        || (criteria.FunctionTolerance > 0.0 && functionDelta <= criteria.FunctionTolerance);
}

} // namespace

FirstOrderMethod ParseFirstOrderMethod(std::string_view methodName)
{
    if (methodName == "gradient_descent" || methodName == "gd" || methodName == "GradientDescent") {
        return FirstOrderMethod::GradientDescent;
    }
    if (methodName == "heavy_ball" || methodName == "momentum" || methodName == "HeavyBall") {
        return FirstOrderMethod::HeavyBall;
    }
    if (methodName == "nesterov" || methodName == "nag" || methodName == "Nesterov") {
        return FirstOrderMethod::Nesterov;
    }
    if (methodName == "adam" || methodName == "Adam") {
        return FirstOrderMethod::Adam;
    }
    if (methodName == "rmsprop" || methodName == "RMSProp") {
        return FirstOrderMethod::RMSProp;
    }
    if (methodName == "adagrad" || methodName == "Adagrad") {
        return FirstOrderMethod::Adagrad;
    }
    throw std::invalid_argument("Unknown first-order optimizer method");
}

OptimizeResult MinimizeFirstOrder(const ValueFunction& valueFunction,
                                  const GradientFunction& gradientFunction,
                                  std::span<const double> x0,
                                  FirstOrderMethod method,
                                  const FirstOrderConfig& config)
{
    auto dimension = x0.size();
    if (dimension == 0) {
        throw std::invalid_argument("Initial point dimension must be positive");
    }
    if (config.LearningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }

    auto start = std::chrono::steady_clock::now();
    Vector current(x0);
    Vector next(dimension);
    Vector gradient(dimension);
    Vector nextGradient(dimension);
    Vector lookahead(dimension);
    Vector velocity(dimension);
    Vector firstMoment(dimension);
    Vector secondMoment(dimension);
    Vector accumulator(dimension);
    Vector delta(dimension);

    auto currentValue = valueFunction(current.Span());
    gradientFunction(current.Span(), gradient.Span());
    auto gradientNorm = Norm2(gradient);

    OptimizeResult result;
    result.Path = Trajectory(dimension);
    if (config.StoreTrajectory) {
        result.Path.Reserve(config.Criteria.MaxIterations + 1);
        result.Path.Record(current.Span(), currentValue, gradientNorm, ElapsedMs(start));
    }

    if (gradientNorm <= config.Criteria.GradientTolerance) {
        result.X = current;
        result.Value = currentValue;
        result.GradientNorm = gradientNorm;
        result.Converged = true;
        return result;
    }

    for (std::size_t iteration = 1; iteration <= config.Criteria.MaxIterations; ++iteration) {
        switch (method) {
        case FirstOrderMethod::GradientDescent:
            for (std::size_t index = 0; index < dimension; ++index) {
                next[index] = current[index] - config.LearningRate * gradient[index];
            }
            break;
        case FirstOrderMethod::HeavyBall:
            for (std::size_t index = 0; index < dimension; ++index) {
                velocity[index] =
                    config.Momentum * velocity[index] - config.LearningRate * gradient[index];
                next[index] = current[index] + velocity[index];
            }
            break;
        case FirstOrderMethod::Nesterov:
            for (std::size_t index = 0; index < dimension; ++index) {
                lookahead[index] = current[index] + config.Momentum * velocity[index];
            }
            gradientFunction(lookahead.Span(), nextGradient.Span());
            for (std::size_t index = 0; index < dimension; ++index) {
                velocity[index] =
                    config.Momentum * velocity[index] - config.LearningRate * nextGradient[index];
                next[index] = current[index] + velocity[index];
            }
            break;
        case FirstOrderMethod::Adam:
            for (std::size_t index = 0; index < dimension; ++index) {
                firstMoment[index] =
                    config.Beta1 * firstMoment[index] + (1.0 - config.Beta1) * gradient[index];
                secondMoment[index] = config.Beta2 * secondMoment[index]
                    + (1.0 - config.Beta2) * gradient[index] * gradient[index];
                auto firstCorrection = 1.0 - std::pow(config.Beta1, static_cast<double>(iteration));
                auto secondCorrection =
                    1.0 - std::pow(config.Beta2, static_cast<double>(iteration));
                auto correctedFirst = firstMoment[index] / firstCorrection;
                auto correctedSecond = secondMoment[index] / secondCorrection;
                next[index] = current[index]
                    - config.LearningRate * correctedFirst
                        / (std::sqrt(correctedSecond) + config.Epsilon);
            }
            break;
        case FirstOrderMethod::RMSProp:
            for (std::size_t index = 0; index < dimension; ++index) {
                secondMoment[index] = config.Beta2 * secondMoment[index]
                    + (1.0 - config.Beta2) * gradient[index] * gradient[index];
                next[index] = current[index]
                    - config.LearningRate * gradient[index]
                        / (std::sqrt(secondMoment[index]) + config.Epsilon);
            }
            break;
        case FirstOrderMethod::Adagrad:
            for (std::size_t index = 0; index < dimension; ++index) {
                accumulator[index] += gradient[index] * gradient[index];
                next[index] = current[index]
                    - config.LearningRate * gradient[index]
                        / (std::sqrt(accumulator[index]) + config.Epsilon);
            }
            break;
        }

        auto nextValue = valueFunction(next.Span());
        gradientFunction(next.Span(), nextGradient.Span());
        auto nextGradientNorm = Norm2(nextGradient);
        Subtract(next.Span(), current.Span(), delta.Span());
        auto stepNorm = Norm2(delta);
        auto functionDelta = std::abs(currentValue - nextValue);

        if (config.StoreTrajectory) {
            result.Path.Record(next.Span(), nextValue, nextGradientNorm, ElapsedMs(start));
        }

        current = next;
        gradient = nextGradient;
        currentValue = nextValue;
        gradientNorm = nextGradientNorm;
        result.Iterations = iteration;

        if (ShouldStop(gradientNorm, stepNorm, functionDelta, config.Criteria)) {
            result.Converged = true;
            break;
        }
    }

    result.X = current;
    result.Value = currentValue;
    result.GradientNorm = gradientNorm;
    return result;
}

OptimizeResult MinimizeRosenbrock(std::span<const double> x0,
                                  FirstOrderMethod method,
                                  const FirstOrderConfig& config)
{
    auto valueFunction = [](std::span<const double> values) {
        return RosenbrockValue(values);
    };
    auto gradientFunction = [](std::span<const double> values, std::span<double> gradient) {
        Rosenbrock(values.size()).Gradient(values, gradient);
    };
    return MinimizeFirstOrder(valueFunction, gradientFunction, x0, method, config);
}

} // namespace optlib
