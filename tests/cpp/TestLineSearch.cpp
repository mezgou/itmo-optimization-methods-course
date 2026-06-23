#include <gtest/gtest.h>

#include <optlib/core/optimizers/LineSearch.hpp>

namespace {

double QuadraticValue(std::span<const double> values)
{
    return (values[0] - 3.0) * (values[0] - 3.0) + 0.5 * (values[1] + 2.0) * (values[1] + 2.0);
}

void QuadraticGradient(std::span<const double> values, std::span<double> gradient)
{
    gradient[0] = 2.0 * (values[0] - 3.0);
    gradient[1] = values[1] + 2.0;
}

} // namespace

TEST(LineSearchTest, ArmijoFindsDecreasingStep)
{
    optlib::Vector x{0.0, 0.0};
    optlib::Vector gradient(2);
    QuadraticGradient(x.Span(), gradient.Span());
    optlib::Vector direction{-gradient[0], -gradient[1]};

    const auto step = optlib::ArmijoLineSearch(
        QuadraticValue, x.Span(), direction.Span(), gradient.Span(), QuadraticValue(x.Span()));
    optlib::Vector next(2);
    next[0] = x[0] + step * direction[0];
    next[1] = x[1] + step * direction[1];

    EXPECT_GT(step, 0.0);
    EXPECT_LT(QuadraticValue(next.Span()), QuadraticValue(x.Span()));
}

TEST(LineSearchTest, WolfeAcceptsDescentStep)
{
    optlib::Vector x{0.0, 0.0};
    optlib::Vector gradient(2);
    QuadraticGradient(x.Span(), gradient.Span());
    optlib::Vector direction{-gradient[0], -gradient[1]};

    const auto step = optlib::WolfeLineSearch(QuadraticValue,
                                              QuadraticGradient,
                                              x.Span(),
                                              direction.Span(),
                                              gradient.Span(),
                                              QuadraticValue(x.Span()));

    EXPECT_GT(step, 0.0);
}
