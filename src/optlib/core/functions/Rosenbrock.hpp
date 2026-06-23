#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>

#include <optlib/core/LinAlg.hpp>

namespace optlib {

enum class DifferentiationScheme;

class Rosenbrock {
public:
    explicit Rosenbrock(std::size_t dimension = 2);

    [[nodiscard]] std::size_t Dimension() const noexcept { return m_Dimension; }

    [[nodiscard]] double Value(std::span<const double> x) const;

    void Gradient(std::span<const double> x, std::span<double> gradient) const;

    [[nodiscard]] Vector Gradient(std::span<const double> x) const;

    void Hessian(std::span<const double> x, Matrix& hessian) const;

    [[nodiscard]] Matrix Hessian(std::span<const double> x) const;

    template <class Scalar>
    [[nodiscard]] Scalar Eval(std::span<const Scalar> x) const
    {
        ValidateDimension(x.size());
        auto result = Scalar{0.0};
        for (std::size_t index = 0; index + 1 < x.size(); ++index) {
            auto residual = x[index + 1] - x[index] * x[index];
            auto linear = Scalar{1.0} - x[index];
            result = result + Scalar{100.0} * residual * residual + linear * linear;
        }
        return result;
    }

private:
    std::size_t m_Dimension;

    void ValidateDimension(std::size_t dimension) const;
};

[[nodiscard]] double RosenbrockValue(std::span<const double> x);

[[nodiscard]] Vector RosenbrockGradient(std::span<const double> x);

[[nodiscard]] Matrix RosenbrockHessian(std::span<const double> x);

[[nodiscard]] Vector RosenbrockAutogradGradient(std::span<const double> x);

[[nodiscard]] Vector RosenbrockNumericalGradient(std::span<const double> x,
                                                 DifferentiationScheme scheme);

} // namespace optlib
