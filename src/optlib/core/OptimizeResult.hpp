#pragma once

#include <chrono>
#include <cstddef>
#include <span>
#include <vector>

#include <optlib/core/LinAlg.hpp>

namespace optlib {

class Trajectory {
public:
    explicit Trajectory(std::size_t dimension = 0) : m_Dimension(dimension) {}

    void Reserve(std::size_t points)
    {
        m_Points.reserve(points * m_Dimension);
        m_Values.reserve(points);
        m_GradientNorms.reserve(points);
        m_TimesMs.reserve(points);
    }

    void Record(std::span<const double> x, double value, double gradientNorm, double timeMs)
    {
        if (m_Dimension == 0) {
            m_Dimension = x.size();
        }
        m_Points.insert(m_Points.end(), x.begin(), x.end());
        m_Values.push_back(value);
        m_GradientNorms.push_back(gradientNorm);
        m_TimesMs.push_back(timeMs);
    }

    [[nodiscard]] std::size_t Dimension() const noexcept { return m_Dimension; }

    [[nodiscard]] std::size_t Size() const noexcept { return m_Values.size(); }

    [[nodiscard]] std::span<const double> Points() const noexcept
    {
        return {m_Points.data(), m_Points.size()};
    }

    [[nodiscard]] std::span<const double> Values() const noexcept
    {
        return {m_Values.data(), m_Values.size()};
    }

    [[nodiscard]] std::span<const double> GradientNorms() const noexcept
    {
        return {m_GradientNorms.data(), m_GradientNorms.size()};
    }

    [[nodiscard]] std::span<const double> TimesMs() const noexcept
    {
        return {m_TimesMs.data(), m_TimesMs.size()};
    }

private:
    std::size_t m_Dimension = 0;
    std::vector<double> m_Points;
    std::vector<double> m_Values;
    std::vector<double> m_GradientNorms;
    std::vector<double> m_TimesMs;
};

struct OptimizeResult {
    Vector X;
    double Value = 0.0;
    double GradientNorm = 0.0;
    std::size_t Iterations = 0;
    bool Converged = false;
    Trajectory Path;
};

[[nodiscard]] inline double ElapsedMs(std::chrono::steady_clock::time_point start)
{
    auto finish = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

} // namespace optlib
