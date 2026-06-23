#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace optlib {

class Dual {
public:
    Dual() = default;

    Dual(double value) : m_Value(value) {}

    Dual(double value, std::size_t dimension) : m_Value(value), m_Gradient(dimension) {}

    Dual(double value, std::vector<double> gradient)
        : m_Value(value), m_Gradient(std::move(gradient))
    {
    }

    [[nodiscard]] static Dual Constant(double value, std::size_t dimension)
    {
        return Dual(value, dimension);
    }

    [[nodiscard]] static Dual Variable(double value, std::size_t variableIndex, std::size_t dimension)
    {
        if (variableIndex >= dimension) {
            throw std::out_of_range("Dual variable index is out of range");
        }

        Dual result(value, dimension);
        result.m_Gradient[variableIndex] = 1.0;
        return result;
    }

    [[nodiscard]] double Value() const noexcept { return m_Value; }

    [[nodiscard]] std::size_t Dimension() const noexcept { return m_Gradient.size(); }

    [[nodiscard]] std::span<const double> Gradient() const noexcept
    {
        return {m_Gradient.data(), m_Gradient.size()};
    }

private:
    double m_Value = 0.0;
    std::vector<double> m_Gradient;

    friend Dual operator+(const Dual& left, const Dual& right);
    friend Dual operator-(const Dual& left, const Dual& right);
    friend Dual operator*(const Dual& left, const Dual& right);
    friend Dual operator/(const Dual& left, const Dual& right);
    friend Dual operator-(const Dual& value);
    friend Dual Sin(const Dual& value);
    friend Dual Cos(const Dual& value);
    friend Dual Tan(const Dual& value);
    friend Dual Exp(const Dual& value);
    friend Dual Log(const Dual& value);
    friend Dual Sqrt(const Dual& value);
    friend Dual Pow(const Dual& value, double exponent);
    friend Dual Pow(const Dual& base, const Dual& exponent);
    friend Dual Abs(const Dual& value);
};

namespace detail {

[[nodiscard]] inline std::size_t ResultDimension(const Dual& left, const Dual& right)
{
    auto leftDimension = left.Dimension();
    auto rightDimension = right.Dimension();
    if (leftDimension == rightDimension) {
        return leftDimension;
    }
    if (leftDimension == 0) {
        return rightDimension;
    }
    if (rightDimension == 0) {
        return leftDimension;
    }
    throw std::invalid_argument("Dual dimensions do not match");
}

[[nodiscard]] inline double GradientAt(const Dual& value, std::size_t index)
{
    if (value.Dimension() == 0) {
        return 0.0;
    }
    return value.Gradient()[index];
}

} // namespace detail

[[nodiscard]] inline Dual operator+(const Dual& left, const Dual& right)
{
    auto dimension = detail::ResultDimension(left, right);
    std::vector<double> gradient(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        gradient[index] = detail::GradientAt(left, index) + detail::GradientAt(right, index);
    }
    return {left.m_Value + right.m_Value, std::move(gradient)};
}

[[nodiscard]] inline Dual operator+(const Dual& left, double right) { return left + Dual(right); }

[[nodiscard]] inline Dual operator+(double left, const Dual& right) { return Dual(left) + right; }

[[nodiscard]] inline Dual operator-(const Dual& left, const Dual& right)
{
    auto dimension = detail::ResultDimension(left, right);
    std::vector<double> gradient(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        gradient[index] = detail::GradientAt(left, index) - detail::GradientAt(right, index);
    }
    return {left.m_Value - right.m_Value, std::move(gradient)};
}

[[nodiscard]] inline Dual operator-(const Dual& left, double right) { return left - Dual(right); }

[[nodiscard]] inline Dual operator-(double left, const Dual& right) { return Dual(left) - right; }

[[nodiscard]] inline Dual operator-(const Dual& value)
{
    std::vector<double> gradient(value.m_Gradient.size());
    for (std::size_t index = 0; index < value.m_Gradient.size(); ++index) {
        gradient[index] = -value.m_Gradient[index];
    }
    return {-value.m_Value, std::move(gradient)};
}

[[nodiscard]] inline Dual operator*(const Dual& left, const Dual& right)
{
    auto dimension = detail::ResultDimension(left, right);
    std::vector<double> gradient(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        gradient[index] = left.m_Value * detail::GradientAt(right, index)
            + right.m_Value * detail::GradientAt(left, index);
    }
    return {left.m_Value * right.m_Value, std::move(gradient)};
}

[[nodiscard]] inline Dual operator*(const Dual& left, double right) { return left * Dual(right); }

[[nodiscard]] inline Dual operator*(double left, const Dual& right) { return Dual(left) * right; }

[[nodiscard]] inline Dual operator/(const Dual& left, const Dual& right)
{
    auto dimension = detail::ResultDimension(left, right);
    auto denominator = right.m_Value * right.m_Value;
    std::vector<double> gradient(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        gradient[index] = (detail::GradientAt(left, index) * right.m_Value
                           - left.m_Value * detail::GradientAt(right, index))
            / denominator;
    }
    return {left.m_Value / right.m_Value, std::move(gradient)};
}

[[nodiscard]] inline Dual operator/(const Dual& left, double right) { return left / Dual(right); }

[[nodiscard]] inline Dual operator/(double left, const Dual& right) { return Dual(left) / right; }

[[nodiscard]] inline Dual Sin(const Dual& value)
{
    auto derivative = std::cos(value.m_Value);
    std::vector<double> gradient(value.m_Gradient.size());
    for (std::size_t index = 0; index < value.m_Gradient.size(); ++index) {
        gradient[index] = derivative * value.m_Gradient[index];
    }
    return {std::sin(value.m_Value), std::move(gradient)};
}

[[nodiscard]] inline Dual Cos(const Dual& value)
{
    auto derivative = -std::sin(value.m_Value);
    std::vector<double> gradient(value.m_Gradient.size());
    for (std::size_t index = 0; index < value.m_Gradient.size(); ++index) {
        gradient[index] = derivative * value.m_Gradient[index];
    }
    return {std::cos(value.m_Value), std::move(gradient)};
}

[[nodiscard]] inline Dual Tan(const Dual& value)
{
    auto tanValue = std::tan(value.m_Value);
    auto derivative = 1.0 + tanValue * tanValue;
    std::vector<double> gradient(value.m_Gradient.size());
    for (std::size_t index = 0; index < value.m_Gradient.size(); ++index) {
        gradient[index] = derivative * value.m_Gradient[index];
    }
    return {tanValue, std::move(gradient)};
}

[[nodiscard]] inline Dual Exp(const Dual& value)
{
    auto expValue = std::exp(value.m_Value);
    std::vector<double> gradient(value.m_Gradient.size());
    for (std::size_t index = 0; index < value.m_Gradient.size(); ++index) {
        gradient[index] = expValue * value.m_Gradient[index];
    }
    return {expValue, std::move(gradient)};
}

[[nodiscard]] inline Dual Log(const Dual& value)
{
    std::vector<double> gradient(value.m_Gradient.size());
    for (std::size_t index = 0; index < value.m_Gradient.size(); ++index) {
        gradient[index] = value.m_Gradient[index] / value.m_Value;
    }
    return {std::log(value.m_Value), std::move(gradient)};
}

[[nodiscard]] inline Dual Sqrt(const Dual& value)
{
    auto sqrtValue = std::sqrt(value.m_Value);
    auto derivative = 0.5 / sqrtValue;
    std::vector<double> gradient(value.m_Gradient.size());
    for (std::size_t index = 0; index < value.m_Gradient.size(); ++index) {
        gradient[index] = derivative * value.m_Gradient[index];
    }
    return {sqrtValue, std::move(gradient)};
}

[[nodiscard]] inline Dual Pow(const Dual& value, double exponent)
{
    auto powValue = std::pow(value.m_Value, exponent);
    auto derivative = exponent * std::pow(value.m_Value, exponent - 1.0);
    std::vector<double> gradient(value.m_Gradient.size());
    for (std::size_t index = 0; index < value.m_Gradient.size(); ++index) {
        gradient[index] = derivative * value.m_Gradient[index];
    }
    return {powValue, std::move(gradient)};
}

[[nodiscard]] inline Dual Pow(const Dual& base, const Dual& exponent)
{
    return Exp(exponent * Log(base));
}

[[nodiscard]] inline Dual Abs(const Dual& value)
{
    auto sign = value.m_Value < 0.0 ? -1.0 : 1.0;
    std::vector<double> gradient(value.m_Gradient.size());
    for (std::size_t index = 0; index < value.m_Gradient.size(); ++index) {
        gradient[index] = sign * value.m_Gradient[index];
    }
    return {std::abs(value.m_Value), std::move(gradient)};
}

[[nodiscard]] inline std::vector<Dual> MakeDualVariables(std::span<const double> values)
{
    std::vector<Dual> variables;
    variables.reserve(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        variables.push_back(Dual::Variable(values[index], index, values.size()));
    }
    return variables;
}

} // namespace optlib
