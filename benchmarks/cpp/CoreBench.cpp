#include <optlib/core/Differentiation.hpp>
#include <optlib/core/LinAlg.hpp>
#include <optlib/core/functions/Rosenbrock.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

template <class Function>
double MeasureMicroseconds(Function&& function, int repeats)
{
    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        auto start = std::chrono::steady_clock::now();
        function();
        auto finish = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double, std::micro>(finish - start).count();
        timings.push_back(elapsed);
    }
    std::sort(timings.begin(), timings.end());
    return timings[timings.size() / 2];
}

void PrintResult(const std::string& name, double microseconds)
{
    std::cout << name << "," << microseconds << '\n';
}

} // namespace

int main()
{
    constexpr int REPEATS = 25;
    constexpr std::size_t DIMENSION = 256;

    optlib::Vector left(DIMENSION);
    optlib::Vector right(DIMENSION);
    for (std::size_t index = 0; index < DIMENSION; ++index) {
        left[index] = 1.0 + static_cast<double>(index % 13);
        right[index] = -0.5 + static_cast<double>(index % 7);
    }

    optlib::Matrix matrix(DIMENSION, DIMENSION);
    for (std::size_t row = 0; row < DIMENSION; ++row) {
        for (std::size_t col = 0; col < DIMENSION; ++col) {
            matrix.At(row, col) = static_cast<double>((row + col) % 11) * 0.01;
        }
    }

    auto function = [](std::span<const double> values) {
        return optlib::RosenbrockValue(values);
    };
    auto dualFunction = [](std::span<const optlib::Dual> values) {
        return optlib::Rosenbrock(values.size()).Eval<optlib::Dual>(values);
    };

    std::cout << "benchmark,median_us\n";
    PrintResult("Dot256x1000",
                MeasureMicroseconds(
                    [&]() {
                        auto value = 0.0;
                        for (int iteration = 0; iteration < 1000; ++iteration) {
                            value += optlib::Dot(left, right);
                        }
                        (void)value;
                    },
                    REPEATS));
    PrintResult("Gemv256",
                MeasureMicroseconds(
                    [&]() {
                        auto result = optlib::Gemv(matrix, left);
                        (void)result;
                    },
                    REPEATS));
    PrintResult("CentralGradient256",
                MeasureMicroseconds(
                    [&]() {
                        auto result = optlib::ComputeGradient(
                            function,
                            left.Span(),
                            optlib::DifferentiationOptions{
                                .Scheme = optlib::DifferentiationScheme::Central});
                        (void)result;
                    },
                    REPEATS));
    PrintResult("AutogradGradient256",
                MeasureMicroseconds(
                    [&]() {
                        auto result = optlib::ComputeAutogradGradient(dualFunction, left.Span());
                        (void)result;
                    },
                    REPEATS));
    return 0;
}
