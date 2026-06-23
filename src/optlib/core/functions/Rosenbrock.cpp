#include <optlib/core/functions/Rosenbrock.hpp>

#include <optlib/core/Differentiation.hpp>

#include <algorithm>

namespace optlib {

Rosenbrock::Rosenbrock(std::size_t dimension) : m_Dimension(dimension)
{
    ValidateDimension(dimension);
}

void Rosenbrock::ValidateDimension(std::size_t dimension) const
{
    if (dimension < 2) {
        throw std::invalid_argument("Rosenbrock dimension must be at least 2");
    }
    if (dimension != m_Dimension) {
        throw std::invalid_argument("Rosenbrock input dimension does not match objective dimension");
    }
}

double Rosenbrock::Value(std::span<const double> x) const
{
    return Eval<double>(x);
}

void Rosenbrock::Gradient(std::span<const double> x, std::span<double> gradient) const
{
    ValidateDimension(x.size());
    if (x.size() != m_Dimension || gradient.size() != m_Dimension) {
        throw std::invalid_argument("Rosenbrock gradient dimension mismatch");
    }

    std::fill(gradient.begin(), gradient.end(), 0.0);
    for (std::size_t index = 0; index + 1 < x.size(); ++index) {
        auto residual = x[index + 1] - x[index] * x[index];
        gradient[index] += -400.0 * x[index] * residual - 2.0 * (1.0 - x[index]);
        gradient[index + 1] += 200.0 * residual;
    }
}

Vector Rosenbrock::Gradient(std::span<const double> x) const
{
    Vector gradient(m_Dimension);
    Gradient(x, gradient.Span());
    return gradient;
}

void Rosenbrock::Hessian(std::span<const double> x, Matrix& hessian) const
{
    ValidateDimension(x.size());
    if (x.size() != m_Dimension) {
        throw std::invalid_argument("Rosenbrock Hessian dimension mismatch");
    }
    if (hessian.Rows() != m_Dimension || hessian.Cols() != m_Dimension) {
        hessian.Resize(m_Dimension, m_Dimension);
    }

    hessian.Fill(0.0);
    for (std::size_t index = 0; index + 1 < x.size(); ++index) {
        hessian.At(index, index) +=
            1200.0 * x[index] * x[index] - 400.0 * x[index + 1] + 2.0;
        hessian.At(index, index + 1) += -400.0 * x[index];
        hessian.At(index + 1, index) += -400.0 * x[index];
        hessian.At(index + 1, index + 1) += 200.0;
    }
}

Matrix Rosenbrock::Hessian(std::span<const double> x) const
{
    Matrix hessian(m_Dimension, m_Dimension);
    Hessian(x, hessian);
    return hessian;
}

double RosenbrockValue(std::span<const double> x)
{
    return Rosenbrock(x.size()).Value(x);
}

Vector RosenbrockGradient(std::span<const double> x)
{
    return Rosenbrock(x.size()).Gradient(x);
}

Matrix RosenbrockHessian(std::span<const double> x)
{
    return Rosenbrock(x.size()).Hessian(x);
}

Vector RosenbrockAutogradGradient(std::span<const double> x)
{
    Rosenbrock function(x.size());
    return ComputeAutogradGradient(
        [&function](std::span<const Dual> dualValues) { return function.Eval<Dual>(dualValues); },
        x);
}

Vector RosenbrockNumericalGradient(std::span<const double> x, DifferentiationScheme scheme)
{
    return ComputeGradient([](std::span<const double> values) { return RosenbrockValue(values); },
                           x, DifferentiationOptions{scheme, 0.0});
}

} // namespace optlib
