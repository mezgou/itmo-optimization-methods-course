#include <gtest/gtest.h>

#include <optlib/core/Version.hpp>

TEST(SmokeTest, VersionIsAvailable)
{
    EXPECT_EQ(optlib::Version(), "1.0.0");
}

TEST(SmokeTest, AddReturnsSum)
{
    EXPECT_DOUBLE_EQ(optlib::Add(2.0, 3.0), 5.0);
}
