#include <optlib/core/optimizers/SecondOrder.hpp>

#include <optlib/core/functions/Rosenbrock.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <vector>

namespace optlib {

namespace {

void AddScaled(std::span<const double> x,
               double alpha,
               std::span<const double> direction,
               std::span<double> result)
{
    for (std::size_t index = 0; index < x.size(); ++index) {
        result[index] = x[index] + alpha * direction[index];
    }
}

void Difference(std::span<const double> left,
                std::span<const double> right,
                std::span<double> result)
{
    for (std::size_t index = 0; index < left.size(); ++index) {
        result[index] = left[index] - right[index];
    }
}

void MatVec(const Matrix& matrix, std::span<const double> vector, std::span<double> result)
{
    for (std::size_t row = 0; row < matrix.Rows(); ++row) {
        auto sum = 0.0;
        for (std::size_t col = 0; col < matrix.Cols(); ++col) {
            sum += matrix.At(row, col) * vector[col];
        }
        result[row] = sum;
    }
}

bool SolveLinearSystem(Matrix matrix, std::span<const double> rhs, std::span<double> result)
{
    auto dimension = rhs.size();
    Vector b(rhs);
    for (std::size_t col = 0; col < dimension; ++col) {
        auto pivotRow = col;
        auto pivotAbs = std::abs(matrix.At(col, col));
        for (std::size_t row = col + 1; row < dimension; ++row) {
            auto candidateAbs = std::abs(matrix.At(row, col));
            if (candidateAbs > pivotAbs) {
                pivotAbs = candidateAbs;
                pivotRow = row;
            }
        }
        if (pivotAbs < 1e-14) {
            return false;
        }
        if (pivotRow != col) {
            for (std::size_t inner = col; inner < dimension; ++inner) {
                std::swap(matrix.At(col, inner), matrix.At(pivotRow, inner));
            }
            std::swap(b[col], b[pivotRow]);
        }
        auto pivot = matrix.At(col, col);
        for (std::size_t row = col + 1; row < dimension; ++row) {
            auto factor = matrix.At(row, col) / pivot;
            matrix.At(row, col) = 0.0;
            for (std::size_t inner = col + 1; inner < dimension; ++inner) {
                matrix.At(row, inner) -= factor * matrix.At(col, inner);
            }
            b[row] -= factor * b[col];
        }
    }

    for (std::size_t offset = 0; offset < dimension; ++offset) {
        auto row = dimension - 1 - offset;
        auto sum = b[row];
        for (std::size_t col = row + 1; col < dimension; ++col) {
            sum -= matrix.At(row, col) * result[col];
        }
        result[row] = sum / matrix.At(row, row);
    }
    return true;
}

void SetIdentity(Matrix& matrix)
{
    matrix.Fill(0.0);
    for (std::size_t index = 0; index < matrix.Rows(); ++index) {
        matrix.At(index, index) = 1.0;
    }
}

bool Stop(double gradientNorm, const StopCriteria& criteria)
{
    return gradientNorm <= criteria.GradientTolerance;
}

} // namespace

SecondOrderMethod ParseSecondOrderMethod(std::string_view methodName)
{
    if (methodName == "newton" || methodName == "Newton") {
        return SecondOrderMethod::Newton;
    }
    if (methodName == "bfgs" || methodName == "BFGS") {
        return SecondOrderMethod::BFGS;
    }
    if (methodName == "lbfgs" || methodName == "l-bfgs" || methodName == "L-BFGS") {
        return SecondOrderMethod::LBFGS;
    }
    throw std::invalid_argument("Unknown second-order method");
}

OptimizeResult MinimizeSecondOrder(const ValueFunction& valueFunction,
                                   const GradientFunction& gradientFunction,
                                   const HessianFunction& hessianFunction,
                                   std::span<const double> x0,
                                   SecondOrderMethod method,
                                   const SecondOrderConfig& config)
{
    auto dimension = x0.size();
    auto start = std::chrono::steady_clock::now();
    Vector current(x0);
    Vector next(dimension);
    Vector gradient(dimension);
    Vector nextGradient(dimension);
    Vector direction(dimension);
    Vector rhs(dimension);
    Vector sVector(dimension);
    Vector yVector(dimension);
    Vector scratch(dimension);
    Matrix hessian(dimension, dimension);
    Matrix inverseHessian(dimension, dimension);
    SetIdentity(inverseHessian);
    std::deque<Vector> sHistory;
    std::deque<Vector> yHistory;
    std::deque<double> rhoHistory;

    auto currentValue = valueFunction(current.Span());
    gradientFunction(current.Span(), gradient.Span());
    auto gradientNorm = Norm2(gradient);

    OptimizeResult result;
    result.Path = Trajectory(dimension);
    if (config.StoreTrajectory) {
        result.Path.Reserve(config.Criteria.MaxIterations + 1);
        result.Path.Record(current.Span(), currentValue, gradientNorm, ElapsedMs(start));
    }

    for (std::size_t iteration = 1; iteration <= config.Criteria.MaxIterations; ++iteration) {
        if (Stop(gradientNorm, config.Criteria)) {
            result.Converged = true;
            break;
        }

        if (method == SecondOrderMethod::Newton) {
            hessianFunction(current.Span(), hessian);
            for (std::size_t index = 0; index < dimension; ++index) {
                hessian.At(index, index) += config.HessianDamping;
                rhs[index] = -gradient[index];
            }
            if (!SolveLinearSystem(hessian, rhs.Span(), direction.Span())) {
                for (std::size_t index = 0; index < dimension; ++index) {
                    direction[index] = -gradient[index];
                }
            }
        } else if (method == SecondOrderMethod::BFGS) {
            MatVec(inverseHessian, gradient.Span(), direction.Span());
            Scale(-1.0, direction.Span());
        } else {
            scratch = gradient;
            std::vector<double> alphaValues(sHistory.size());
            for (std::size_t offset = 0; offset < sHistory.size(); ++offset) {
                auto index = sHistory.size() - 1 - offset;
                auto alpha = rhoHistory[index] * Dot(sHistory[index], scratch);
                alphaValues[index] = alpha;
                Axpy(-alpha, yHistory[index].Span(), scratch.Span());
            }
            auto scale = 1.0;
            if (!sHistory.empty()) {
                auto last = sHistory.size() - 1;
                auto yy = Dot(yHistory[last], yHistory[last]);
                if (yy > 0.0) {
                    scale = Dot(sHistory[last], yHistory[last]) / yy;
                }
            }
            Scale(scale, scratch.Span());
            for (std::size_t index = 0; index < sHistory.size(); ++index) {
                auto beta = rhoHistory[index] * Dot(yHistory[index], scratch);
                Axpy(alphaValues[index] - beta, sHistory[index].Span(), scratch.Span());
            }
            direction = scratch;
            Scale(-1.0, direction.Span());
        }

        auto step = method == SecondOrderMethod::BFGS
            ? WolfeLineSearch(
                valueFunction, gradientFunction, current.Span(), direction.Span(), gradient.Span(), currentValue, config.Search)
            : ArmijoLineSearch(
                valueFunction, current.Span(), direction.Span(), gradient.Span(), currentValue, config.Search);
        if (step <= 0.0) {
            for (std::size_t index = 0; index < dimension; ++index) {
                direction[index] = -gradient[index];
            }
            step = ArmijoLineSearch(
                valueFunction, current.Span(), direction.Span(), gradient.Span(), currentValue, config.Search);
        }

        AddScaled(current.Span(), step, direction.Span(), next.Span());
        auto nextValue = valueFunction(next.Span());
        gradientFunction(next.Span(), nextGradient.Span());
        Difference(next.Span(), current.Span(), sVector.Span());
        Difference(nextGradient.Span(), gradient.Span(), yVector.Span());
        auto yDotS = Dot(yVector, sVector);

        if (method == SecondOrderMethod::BFGS && yDotS > 1e-14) {
            auto rho = 1.0 / yDotS;
            MatVec(inverseHessian, yVector.Span(), scratch.Span());
            auto yDotHy = Dot(yVector, scratch);
            Matrix updated(dimension, dimension);
            for (std::size_t row = 0; row < dimension; ++row) {
                for (std::size_t col = 0; col < dimension; ++col) {
                    auto value = inverseHessian.At(row, col);
                    value += (rho + rho * rho * yDotHy) * sVector[row] * sVector[col];
                    value -= rho * (sVector[row] * scratch[col] + scratch[row] * sVector[col]);
                    updated.At(row, col) = value;
                }
            }
            inverseHessian = updated;
        } else if (method == SecondOrderMethod::LBFGS && yDotS > 1e-14) {
            if (sHistory.size() == config.HistorySize) {
                sHistory.pop_front();
                yHistory.pop_front();
                rhoHistory.pop_front();
            }
            sHistory.push_back(sVector);
            yHistory.push_back(yVector);
            rhoHistory.push_back(1.0 / yDotS);
        }

        current = next;
        gradient = nextGradient;
        currentValue = nextValue;
        gradientNorm = Norm2(gradient);
        result.Iterations = iteration;
        if (config.StoreTrajectory) {
            result.Path.Record(current.Span(), currentValue, gradientNorm, ElapsedMs(start));
        }
    }

    result.X = current;
    result.Value = currentValue;
    result.GradientNorm = gradientNorm;
    if (gradientNorm <= config.Criteria.GradientTolerance) {
        result.Converged = true;
    }
    return result;
}

OptimizeResult MinimizeRosenbrockSecondOrder(std::span<const double> x0,
                                             SecondOrderMethod method,
                                             const SecondOrderConfig& config)
{
    auto valueFunction = [](std::span<const double> values) {
        return RosenbrockValue(values);
    };
    auto gradientFunction = [](std::span<const double> values, std::span<double> gradient) {
        Rosenbrock(values.size()).Gradient(values, gradient);
    };
    auto hessianFunction = [](std::span<const double> values, Matrix& hessian) {
        Rosenbrock(values.size()).Hessian(values, hessian);
    };
    return MinimizeSecondOrder(valueFunction, gradientFunction, hessianFunction, x0, method, config);
}

} // namespace optlib
