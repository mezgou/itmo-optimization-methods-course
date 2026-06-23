#pragma once

#include <string_view>

namespace optlib {

inline constexpr std::string_view VERSION = "1.0.0";

[[nodiscard]] inline constexpr std::string_view Version() noexcept { return VERSION; }

[[nodiscard]] inline constexpr double Add(double leftValue, double rightValue) noexcept
{
    return leftValue + rightValue;
}

} // namespace optlib
