#include <optlib/core/optimizers/ZeroOrder.hpp>

#include <optlib/core/functions/Rosenbrock.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace optlib {

namespace {

constexpr double GOLDEN_RATIO_INV = 0.6180339887498948482;

struct LineMinimum {
    Vector X;
    double Value = 0.0;
    std::size_t Evaluations = 0;
};

void AddScaled(std::span<const double> left,
               double scale,
               std::span<const double> direction,
               std::span<double> result)
{
    if (left.size() != direction.size() || left.size() != result.size()) {
        throw std::invalid_argument("AddScaled sizes do not match");
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        result[index] = left[index] + scale * direction[index];
    }
}

void Subtract(std::span<const double> left, std::span<const double> right, std::span<double> result)
{
    if (left.size() != right.size() || left.size() != result.size()) {
        throw std::invalid_argument("Subtract sizes do not match");
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        result[index] = left[index] - right[index];
    }
}

[[nodiscard]] bool ShouldStop(double stepNorm, double functionDelta, const StopCriteria& criteria)
{
    return (criteria.StepTolerance > 0.0 && stepNorm <= criteria.StepTolerance)
        || (criteria.FunctionTolerance > 0.0 && functionDelta <= criteria.FunctionTolerance);
}

LineMinimum GoldenSectionSearch(const ValueFunction& valueFunction,
                                std::span<const double> x,
                                std::span<const double> direction,
                                double radius,
                                const ZeroOrderConfig& config)
{
    auto left = -radius;
    auto right = radius;
    auto c = right - GOLDEN_RATIO_INV * (right - left);
    auto d = left + GOLDEN_RATIO_INV * (right - left);
    Vector cPoint(x.size());
    Vector dPoint(x.size());
    AddScaled(x, c, direction, cPoint.Span());
    AddScaled(x, d, direction, dPoint.Span());
    auto cValue = valueFunction(cPoint.Span());
    auto dValue = valueFunction(dPoint.Span());
    std::size_t evaluations = 2;

    for (std::size_t iteration = 0; iteration < config.LineSearchIterations; ++iteration) {
        if (std::abs(right - left) <= config.LineSearchTolerance) {
            break;
        }
        if (cValue < dValue) {
            right = d;
            d = c;
            dValue = cValue;
            dPoint = cPoint;
            c = right - GOLDEN_RATIO_INV * (right - left);
            AddScaled(x, c, direction, cPoint.Span());
            cValue = valueFunction(cPoint.Span());
        } else {
            left = c;
            c = d;
            cValue = dValue;
            cPoint = dPoint;
            d = left + GOLDEN_RATIO_INV * (right - left);
            AddScaled(x, d, direction, dPoint.Span());
            dValue = valueFunction(dPoint.Span());
        }
        ++evaluations;
    }

    if (cValue <= dValue) {
        return {.X = std::move(cPoint), .Value = cValue, .Evaluations = evaluations};
    }
    return {.X = std::move(dPoint), .Value = dValue, .Evaluations = evaluations};
}

OptimizeResult MinimizeNelderMead(const ValueFunction& valueFunction,
                                  std::span<const double> x0,
                                  const ZeroOrderConfig& config)
{
    const auto dimension = x0.size();
    const auto start = std::chrono::steady_clock::now();
    std::vector<Vector> simplex(dimension + 1, Vector(x0));
    std::vector<double> values(dimension + 1);

    OptimizeResult result;
    result.Path = Trajectory(dimension);
    for (std::size_t vertex = 1; vertex <= dimension; ++vertex) {
        const auto coord = vertex - 1;
        simplex[vertex][coord] += config.InitialStep * std::max(1.0, std::abs(x0[coord]));
    }
    for (std::size_t vertex = 0; vertex <= dimension; ++vertex) {
        values[vertex] = valueFunction(simplex[vertex].Span());
        ++result.FunctionEvaluations;
    }

    if (config.StoreTrajectory) {
        result.Path.Reserve(config.Criteria.MaxIterations + 1);
    }

    for (std::size_t iteration = 0; iteration <= config.Criteria.MaxIterations; ++iteration) {
        std::vector<std::size_t> order(dimension + 1);
        std::iota(order.begin(), order.end(), 0);
        std::ranges::sort(order, [&](std::size_t left, std::size_t right) {
            return values[left] < values[right];
        });

        const auto bestIndex = order[0];
        const auto worstIndex = order[dimension];
        const auto secondWorstIndex = order[dimension - 1];
        const auto bestValue = values[bestIndex];
        auto simplexDiameter = 0.0;
        auto valueSpread = 0.0;
        for (std::size_t rank = 1; rank <= dimension; ++rank) {
            Vector distance(dimension);
            Subtract(simplex[order[rank]].Span(), simplex[bestIndex].Span(), distance.Span());
            simplexDiameter = std::max(simplexDiameter, Norm2(distance));
            valueSpread = std::max(valueSpread, std::abs(values[order[rank]] - bestValue));
        }
        if (config.StoreTrajectory) {
            result.Path.Record(simplex[bestIndex].Span(), bestValue, 0.0, ElapsedMs(start));
        }
        if (iteration > 0) {
            const auto smallSimplex =
                config.Criteria.StepTolerance > 0.0 && simplexDiameter <= config.Criteria.StepTolerance;
            const auto flatSimplex =
                config.Criteria.FunctionTolerance > 0.0
                && valueSpread <= config.Criteria.FunctionTolerance;
            if (smallSimplex || flatSimplex) {
                result.Converged = true;
                break;
            }
        }
        if (iteration == config.Criteria.MaxIterations) {
            break;
        }

        Vector centroid(dimension);
        centroid.Fill(0.0);
        for (std::size_t rank = 0; rank < dimension; ++rank) {
            Axpy(1.0 / static_cast<double>(dimension), simplex[order[rank]].Span(), centroid.Span());
        }

        Vector reflected(dimension);
        for (std::size_t coord = 0; coord < dimension; ++coord) {
            reflected[coord] =
                centroid[coord] + config.Reflection * (centroid[coord] - simplex[worstIndex][coord]);
        }
        auto reflectedValue = valueFunction(reflected.Span());
        ++result.FunctionEvaluations;

        if (reflectedValue < bestValue) {
            Vector expanded(dimension);
            for (std::size_t coord = 0; coord < dimension; ++coord) {
                expanded[coord] = centroid[coord]
                    + config.Expansion * (reflected[coord] - centroid[coord]);
            }
            auto expandedValue = valueFunction(expanded.Span());
            ++result.FunctionEvaluations;
            if (expandedValue < reflectedValue) {
                simplex[worstIndex] = std::move(expanded);
                values[worstIndex] = expandedValue;
            } else {
                simplex[worstIndex] = std::move(reflected);
                values[worstIndex] = reflectedValue;
            }
        } else if (reflectedValue < values[secondWorstIndex]) {
            simplex[worstIndex] = std::move(reflected);
            values[worstIndex] = reflectedValue;
        } else {
            Vector contracted(dimension);
            const bool outside = reflectedValue < values[worstIndex];
            const Vector& base = outside ? reflected : simplex[worstIndex];
            for (std::size_t coord = 0; coord < dimension; ++coord) {
                contracted[coord] =
                    centroid[coord] + config.Contraction * (base[coord] - centroid[coord]);
            }
            auto contractedValue = valueFunction(contracted.Span());
            ++result.FunctionEvaluations;
            if (contractedValue < (outside ? reflectedValue : values[worstIndex])) {
                simplex[worstIndex] = std::move(contracted);
                values[worstIndex] = contractedValue;
            } else {
                for (std::size_t rank = 1; rank <= dimension; ++rank) {
                    auto index = order[rank];
                    for (std::size_t coord = 0; coord < dimension; ++coord) {
                        simplex[index][coord] = simplex[bestIndex][coord]
                            + config.Shrink * (simplex[index][coord] - simplex[bestIndex][coord]);
                    }
                    values[index] = valueFunction(simplex[index].Span());
                    ++result.FunctionEvaluations;
                }
            }
        }

        result.Iterations = iteration + 1;
    }

    auto bestIndex = std::min_element(values.begin(), values.end()) - values.begin();
    result.X = simplex[static_cast<std::size_t>(bestIndex)];
    result.Value = values[static_cast<std::size_t>(bestIndex)];
    return result;
}

OptimizeResult MinimizeDirectionSet(const ValueFunction& valueFunction,
                                    std::span<const double> x0,
                                    ZeroOrderMethod method,
                                    const ZeroOrderConfig& config)
{
    const auto dimension = x0.size();
    const auto start = std::chrono::steady_clock::now();
    Vector current(x0);
    auto currentValue = valueFunction(current.Span());

    OptimizeResult result;
    ++result.FunctionEvaluations;
    result.Path = Trajectory(dimension);
    if (config.StoreTrajectory) {
        result.Path.Reserve(config.Criteria.MaxIterations + 1);
        result.Path.Record(current.Span(), currentValue, 0.0, ElapsedMs(start));
    }

    std::vector<Vector> directions(dimension, Vector(dimension));
    for (std::size_t index = 0; index < dimension; ++index) {
        directions[index][index] = 1.0;
    }

    for (std::size_t iteration = 1; iteration <= config.Criteria.MaxIterations; ++iteration) {
        Vector previous(current.Span());
        const auto previousValue = currentValue;
        std::size_t maxDecreaseIndex = 0;
        auto maxDecrease = 0.0;

        for (std::size_t directionIndex = 0; directionIndex < dimension; ++directionIndex) {
            const auto before = currentValue;
            auto lineMinimum = GoldenSectionSearch(valueFunction,
                                                   current.Span(),
                                                   directions[directionIndex].Span(),
                                                   config.LineSearchRadius,
                                                   config);
            result.FunctionEvaluations += lineMinimum.Evaluations;
            current = std::move(lineMinimum.X);
            currentValue = lineMinimum.Value;
            const auto decrease = before - currentValue;
            if (decrease > maxDecrease) {
                maxDecrease = decrease;
                maxDecreaseIndex = directionIndex;
            }
        }

        if (method == ZeroOrderMethod::Powell) {
            Vector newDirection(dimension);
            Subtract(current.Span(), previous.Span(), newDirection.Span());
            if (Norm2(newDirection) > 0.0) {
                auto lineMinimum = GoldenSectionSearch(valueFunction,
                                                       current.Span(),
                                                       newDirection.Span(),
                                                       config.LineSearchRadius,
                                                       config);
                result.FunctionEvaluations += lineMinimum.Evaluations;
                current = std::move(lineMinimum.X);
                currentValue = lineMinimum.Value;
                directions[maxDecreaseIndex] = std::move(newDirection);
            }
        }

        Vector step(dimension);
        Subtract(current.Span(), previous.Span(), step.Span());
        const auto stepNorm = Norm2(step);
        const auto functionDelta = std::abs(previousValue - currentValue);
        result.Iterations = iteration;
        if (config.StoreTrajectory) {
            result.Path.Record(current.Span(), currentValue, 0.0, ElapsedMs(start));
        }
        if (ShouldStop(stepNorm, functionDelta, config.Criteria)) {
            result.Converged = true;
            break;
        }
    }

    result.X = current;
    result.Value = currentValue;
    return result;
}

} // namespace

ZeroOrderMethod ParseZeroOrderMethod(std::string_view methodName)
{
    if (methodName == "nelder_mead" || methodName == "nelder-mead"
        || methodName == "NelderMead") {
        return ZeroOrderMethod::NelderMead;
    }
    if (methodName == "powell" || methodName == "Powell") {
        return ZeroOrderMethod::Powell;
    }
    if (methodName == "coordinate" || methodName == "coordinate_search"
        || methodName == "coordinate_descent"
        || methodName == "CoordinateSearch") {
        return ZeroOrderMethod::CoordinateSearch;
    }
    throw std::invalid_argument("Unknown zero-order optimizer method");
}

OptimizeResult MinimizeZeroOrder(const ValueFunction& valueFunction,
                                 std::span<const double> x0,
                                 ZeroOrderMethod method,
                                 const ZeroOrderConfig& config)
{
    if (x0.empty()) {
        throw std::invalid_argument("Initial point dimension must be positive");
    }
    if (config.InitialStep <= 0.0 || config.LineSearchRadius <= 0.0) {
        throw std::invalid_argument("Zero-order step sizes must be positive");
    }
    if (method == ZeroOrderMethod::NelderMead) {
        return MinimizeNelderMead(valueFunction, x0, config);
    }
    return MinimizeDirectionSet(valueFunction, x0, method, config);
}

OptimizeResult MinimizeRosenbrockZeroOrder(std::span<const double> x0,
                                           ZeroOrderMethod method,
                                           const ZeroOrderConfig& config)
{
    auto valueFunction = [](std::span<const double> values) {
        return RosenbrockValue(values);
    };
    return MinimizeZeroOrder(valueFunction, x0, method, config);
}

} // namespace optlib
