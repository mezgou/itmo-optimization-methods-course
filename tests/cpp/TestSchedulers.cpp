#include <gtest/gtest.h>

#include <optlib/core/schedulers/LearningRate.hpp>

#include <cmath>

TEST(SchedulerTest, ComputesStepSchedule)
{
    optlib::LearningRateConfig config;
    config.Schedule = optlib::LearningRateSchedule::Step;
    config.InitialLearningRate = 0.1;
    config.Gamma = 0.5;
    config.StepSize = 3;

    EXPECT_DOUBLE_EQ(optlib::LearningRateAt(0, config), 0.1);
    EXPECT_DOUBLE_EQ(optlib::LearningRateAt(2, config), 0.1);
    EXPECT_DOUBLE_EQ(optlib::LearningRateAt(3, config), 0.05);
    EXPECT_DOUBLE_EQ(optlib::LearningRateAt(6, config), 0.025);
}

TEST(SchedulerTest, ComputesExponentialSchedule)
{
    optlib::LearningRateConfig config;
    config.Schedule = optlib::LearningRateSchedule::Exponential;
    config.InitialLearningRate = 0.2;
    config.DecayRate = 0.1;

    EXPECT_NEAR(optlib::LearningRateAt(5, config), 0.2 * std::exp(-0.5), 1e-15);
}

TEST(SchedulerTest, ComputesCosineScheduleWithWarmup)
{
    optlib::LearningRateConfig config;
    config.Schedule = optlib::LearningRateSchedule::Cosine;
    config.InitialLearningRate = 0.1;
    config.MinimumLearningRate = 0.01;
    config.TotalIterations = 10;
    config.WarmupSteps = 2;

    EXPECT_NEAR(optlib::LearningRateAt(0, config), 0.05, 1e-15);
    EXPECT_NEAR(optlib::LearningRateAt(10, config), 0.01, 1e-15);
}
