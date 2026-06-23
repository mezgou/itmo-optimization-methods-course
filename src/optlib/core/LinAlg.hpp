#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

namespace optlib {

class Vector {
public:
    Vector() = default;

    explicit Vector(std::size_t size) : m_Data(size) {}

    Vector(std::initializer_list<double> values) : m_Data(values) {}

    explicit Vector(std::span<const double> values) : m_Data(values.begin(), values.end()) {}

    [[nodiscard]] std::size_t Size() const noexcept { return m_Data.size(); }

    [[nodiscard]] bool Empty() const noexcept { return m_Data.empty(); }

    [[nodiscard]] double* Data() noexcept { return m_Data.data(); }

    [[nodiscard]] const double* Data() const noexcept { return m_Data.data(); }

    [[nodiscard]] std::span<double> Span() noexcept { return {m_Data.data(), m_Data.size()}; }

    [[nodiscard]] std::span<const double> Span() const noexcept
    {
        return {m_Data.data(), m_Data.size()};
    }

    [[nodiscard]] double& operator[](std::size_t index) noexcept { return m_Data[index]; }

    [[nodiscard]] double operator[](std::size_t index) const noexcept { return m_Data[index]; }

    void Resize(std::size_t size) { m_Data.resize(size); }

    void Fill(double value) { std::fill(m_Data.begin(), m_Data.end(), value); }

    [[nodiscard]] std::vector<double> ToStdVector() const { return m_Data; }

private:
    std::vector<double> m_Data;
};

class Matrix {
public:
    Matrix() = default;

    Matrix(std::size_t rows, std::size_t cols) : m_Rows(rows), m_Cols(cols), m_Data(rows * cols) {}

    Matrix(std::size_t rows, std::size_t cols, std::span<const double> values)
        : m_Rows(rows), m_Cols(cols), m_Data(values.begin(), values.end())
    {
        if (values.size() != rows * cols) {
            throw std::invalid_argument("Matrix values size does not match shape");
        }
    }

    [[nodiscard]] std::size_t Rows() const noexcept { return m_Rows; }

    [[nodiscard]] std::size_t Cols() const noexcept { return m_Cols; }

    [[nodiscard]] std::size_t Size() const noexcept { return m_Data.size(); }

    [[nodiscard]] double* Data() noexcept { return m_Data.data(); }

    [[nodiscard]] const double* Data() const noexcept { return m_Data.data(); }

    [[nodiscard]] std::span<double> Span() noexcept { return {m_Data.data(), m_Data.size()}; }

    [[nodiscard]] std::span<const double> Span() const noexcept
    {
        return {m_Data.data(), m_Data.size()};
    }

    [[nodiscard]] double& At(std::size_t row, std::size_t col) noexcept
    {
        return m_Data[row * m_Cols + col];
    }

    [[nodiscard]] double At(std::size_t row, std::size_t col) const noexcept
    {
        return m_Data[row * m_Cols + col];
    }

    void Resize(std::size_t rows, std::size_t cols)
    {
        m_Rows = rows;
        m_Cols = cols;
        m_Data.resize(rows * cols);
    }

    void Fill(double value) { std::fill(m_Data.begin(), m_Data.end(), value); }

    [[nodiscard]] std::vector<double> ToStdVector() const { return m_Data; }

private:
    std::size_t m_Rows = 0;
    std::size_t m_Cols = 0;
    std::vector<double> m_Data;
};

namespace detail {

inline void RequireSameSize(std::span<const double> left, std::span<const double> right)
{
    if (left.size() != right.size()) {
        throw std::invalid_argument("Vector sizes do not match");
    }
}

} // namespace detail

[[nodiscard]] inline double Dot(std::span<const double> left, std::span<const double> right)
{
    detail::RequireSameSize(left, right);
    auto sum = 0.0;
    auto size = left.size();
    for (std::size_t index = 0; index < size; ++index) {
        sum += left[index] * right[index];
    }
    return sum;
}

[[nodiscard]] inline double Dot(const Vector& left, const Vector& right)
{
    return Dot(left.Span(), right.Span());
}

inline void Copy(std::span<const double> source, std::span<double> target)
{
    if (source.size() != target.size()) {
        throw std::invalid_argument("Copy sizes do not match");
    }
    std::copy(source.begin(), source.end(), target.begin());
}

inline void Axpy(double alpha, std::span<const double> x, std::span<double> y)
{
    detail::RequireSameSize(x, y);
    auto size = x.size();
    for (std::size_t index = 0; index < size; ++index) {
        y[index] += alpha * x[index];
    }
}

inline void Scale(double alpha, std::span<double> values)
{
    for (double& value : values) {
        value *= alpha;
    }
}

[[nodiscard]] inline double Norm2(std::span<const double> values)
{
    return std::sqrt(Dot(values, values));
}

[[nodiscard]] inline double Norm2(const Vector& values) { return Norm2(values.Span()); }

[[nodiscard]] inline double NormInf(std::span<const double> values)
{
    auto maximum = 0.0;
    for (double value : values) {
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

[[nodiscard]] inline double NormInf(const Vector& values) { return NormInf(values.Span()); }

inline void Gemv(const Matrix& matrix, const Vector& vector, Vector& result)
{
    if (matrix.Cols() != vector.Size()) {
        throw std::invalid_argument("Gemv shape mismatch");
    }
    if (result.Size() != matrix.Rows()) {
        result.Resize(matrix.Rows());
    }

    for (std::size_t row = 0; row < matrix.Rows(); ++row) {
        auto sum = 0.0;
        for (std::size_t col = 0; col < matrix.Cols(); ++col) {
            sum += matrix.At(row, col) * vector[col];
        }
        result[row] = sum;
    }
}

[[nodiscard]] inline Vector Gemv(const Matrix& matrix, const Vector& vector)
{
    Vector result(matrix.Rows());
    Gemv(matrix, vector, result);
    return result;
}

inline void Gemm(const Matrix& left, const Matrix& right, Matrix& result)
{
    if (left.Cols() != right.Rows()) {
        throw std::invalid_argument("Gemm shape mismatch");
    }
    if (result.Rows() != left.Rows() || result.Cols() != right.Cols()) {
        result.Resize(left.Rows(), right.Cols());
    }
    result.Fill(0.0);

    for (std::size_t row = 0; row < left.Rows(); ++row) {
        for (std::size_t inner = 0; inner < left.Cols(); ++inner) {
            auto leftValue = left.At(row, inner);
            for (std::size_t col = 0; col < right.Cols(); ++col) {
                result.At(row, col) += leftValue * right.At(inner, col);
            }
        }
    }
}

[[nodiscard]] inline Matrix Gemm(const Matrix& left, const Matrix& right)
{
    Matrix result(left.Rows(), right.Cols());
    Gemm(left, right, result);
    return result;
}

} // namespace optlib
