#include <optlib/core/optimizers/LineSearch.hpp>

#include <cmath>
#include <stdexcept>

namespace optlib {

namespace {

void AddScaled(std::span<const double> x,
               double alpha,
               std::span<const double> direction,
               std::span<double> result)
{
    if (x.size() != direction.size() || x.size() != result.size()) {
        throw std::invalid_argument("Line search vector dimensions do not match");
    }
    for (std::size_t index = 0; index < x.size(); ++index) {
        result[index] = x[index] + alpha * direction[index];
    }
}

} // namespace

LineSearchMethod ParseLineSearchMethod(std::string_view methodName)
{
    if (methodName == "armijo" || methodName == "backtracking" || methodName == "Armijo") {
        return LineSearchMethod::Armijo;
    }
    if (methodName == "strong_wolfe" || methodName == "wolfe" || methodName == "StrongWolfe") {
        return LineSearchMethod::StrongWolfe;
    }
    throw std::invalid_argument("Unknown line-search method");
}

double ArmijoLineSearch(const ValueFunction& valueFunction,
                        std::span<const double> x,
                        std::span<const double> direction,
                        std::span<const double> gradient,
                        double currentValue,
                        const LineSearchConfig& config)
{
    auto directionalDerivative = Dot(gradient, direction);
    if (directionalDerivative >= 0.0) {
        return 0.0;
    }

    Vector candidate(x.size());
    auto step = config.InitialStep;
    for (std::size_t iteration = 0; iteration < config.MaxIterations; ++iteration) {
        AddScaled(x, step, direction, candidate.Span());
        auto candidateValue = valueFunction(candidate.Span());
        if (candidateValue <= currentValue + config.C1 * step * directionalDerivative) {
            return step;
        }
        step *= config.Reduction;
        if (step < config.MinimumStep) {
            break;
        }
    }
    return config.MinimumStep;
}

double WolfeLineSearch(const ValueFunction& valueFunction,
                       const GradientFunction& gradientFunction,
                       std::span<const double> x,
                       std::span<const double> direction,
                       std::span<const double> gradient,
                       double currentValue,
                       const LineSearchConfig& config)
{
    auto initialDerivative = Dot(gradient, direction);
    if (initialDerivative >= 0.0) {
        return 0.0;
    }

    Vector candidate(x.size());
    Vector candidateGradient(x.size());
    auto step = config.InitialStep;
    for (std::size_t iteration = 0; iteration < config.MaxIterations; ++iteration) {
        AddScaled(x, step, direction, candidate.Span());
        auto candidateValue = valueFunction(candidate.Span());
        if (candidateValue <= currentValue + config.C1 * step * initialDerivative) {
            gradientFunction(candidate.Span(), candidateGradient.Span());
            auto candidateDerivative = Dot(candidateGradient.Span(), direction);
            if (std::abs(candidateDerivative) <= config.C2 * std::abs(initialDerivative)) {
                return step;
            }
        }
        step *= config.Reduction;
        if (step < config.MinimumStep) {
            break;
        }
    }
    return ArmijoLineSearch(valueFunction, x, direction, gradient, currentValue, config);
}

} // namespace optlib
