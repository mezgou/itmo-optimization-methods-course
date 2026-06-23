#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <optlib/core/Differentiation.hpp>
#include <optlib/core/LinAlg.hpp>
#include <optlib/core/OptimizeResult.hpp>
#include <optlib/core/Version.hpp>
#include <optlib/core/functions/Rosenbrock.hpp>
#include <optlib/core/optimizers/FirstOrder.hpp>
#include <optlib/core/optimizers/SecondOrder.hpp>
#include <optlib/core/optimizers/ZeroOrder.hpp>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::span<const double> VectorSpan(const Array& array)
{
    py::buffer_info info = array.request();
    if (info.ndim != 1) {
        throw std::invalid_argument("Expected a one-dimensional numpy array");
    }
    return {static_cast<const double*>(info.ptr), static_cast<std::size_t>(info.shape[0])};
}

optlib::Vector VectorFromArray(const Array& array)
{
    return optlib::Vector(VectorSpan(array));
}

optlib::Matrix MatrixFromArray(const Array& array)
{
    py::buffer_info info = array.request();
    if (info.ndim != 2) {
        throw std::invalid_argument("Expected a two-dimensional numpy array");
    }
    auto rows = static_cast<std::size_t>(info.shape[0]);
    auto cols = static_cast<std::size_t>(info.shape[1]);
    auto values = static_cast<const double*>(info.ptr);
    return optlib::Matrix(rows, cols, {values, rows * cols});
}

py::array_t<double> VectorToArray(optlib::Vector values)
{
    auto* storage = new optlib::Vector(std::move(values));
    py::capsule owner(storage, [](void* pointer) {
        delete static_cast<optlib::Vector*>(pointer);
    });
    return py::array_t<double>(
        {static_cast<py::ssize_t>(storage->Size())},
        {static_cast<py::ssize_t>(sizeof(double))},
        storage->Data(),
        owner);
}

py::array_t<double> MatrixToArray(optlib::Matrix values)
{
    auto* storage = new optlib::Matrix(std::move(values));
    py::capsule owner(storage, [](void* pointer) {
        delete static_cast<optlib::Matrix*>(pointer);
    });
    return py::array_t<double>(
        {static_cast<py::ssize_t>(storage->Rows()), static_cast<py::ssize_t>(storage->Cols())},
        {static_cast<py::ssize_t>(sizeof(double) * storage->Cols()),
         static_cast<py::ssize_t>(sizeof(double))},
        storage->Data(),
        owner);
}

py::array_t<double> SpanToVectorArray(std::span<const double> values)
{
    return VectorToArray(optlib::Vector(values));
}

py::array_t<double> SpanToMatrixArray(std::size_t rows,
                                      std::size_t cols,
                                      std::span<const double> values)
{
    return MatrixToArray(optlib::Matrix(rows, cols, values));
}

optlib::DifferentiationScheme ParseScheme(const std::string& scheme)
{
    if (scheme == "forward") {
        return optlib::DifferentiationScheme::Forward;
    }
    if (scheme == "central") {
        return optlib::DifferentiationScheme::Central;
    }
    if (scheme == "five_point" || scheme == "five-point" || scheme == "fivepoint") {
        return optlib::DifferentiationScheme::FivePoint;
    }
    throw std::invalid_argument("Unknown differentiation scheme");
}

optlib::FirstOrderConfig MakeFirstOrderConfig(double learningRate,
                                              std::size_t maxIter,
                                              double gradientTolerance,
                                              double stepTolerance,
                                              double functionTolerance,
                                              double momentum,
                                              double beta1,
                                              double beta2,
                                              double epsilon,
                                              bool logTrajectory,
                                              const std::string& schedule,
                                              double learningRateGamma,
                                              std::size_t learningRateStepSize,
                                              double learningRateDecay,
                                              double minimumLearningRate,
                                              std::size_t warmupSteps,
                                              std::size_t scheduleIterations)
{
    optlib::FirstOrderConfig config;
    config.LearningRate = learningRate;
    config.Momentum = momentum;
    config.Beta1 = beta1;
    config.Beta2 = beta2;
    config.Epsilon = epsilon;
    config.StoreTrajectory = logTrajectory;
    config.Schedule = optlib::ParseLearningRateSchedule(schedule);
    config.LearningRateGamma = learningRateGamma;
    config.LearningRateStepSize = learningRateStepSize;
    config.LearningRateDecay = learningRateDecay;
    config.MinimumLearningRate = minimumLearningRate;
    config.WarmupSteps = warmupSteps;
    config.ScheduleIterations = scheduleIterations;
    config.Criteria.MaxIterations = maxIter;
    config.Criteria.GradientTolerance = gradientTolerance;
    config.Criteria.StepTolerance = stepTolerance;
    config.Criteria.FunctionTolerance = functionTolerance;
    return config;
}

optlib::SecondOrderConfig MakeSecondOrderConfig(std::size_t maxIter,
                                                double gradientTolerance,
                                                double stepTolerance,
                                                double functionTolerance,
                                                std::size_t historySize,
                                                double hessianDamping,
                                                const std::string& lineSearch,
                                                double lineSearchInitialStep,
                                                double lineSearchReduction,
                                                bool logTrajectory)
{
    optlib::SecondOrderConfig config;
    config.Criteria.MaxIterations = maxIter;
    config.Criteria.GradientTolerance = gradientTolerance;
    config.Criteria.StepTolerance = stepTolerance;
    config.Criteria.FunctionTolerance = functionTolerance;
    config.HistorySize = historySize;
    config.HessianDamping = hessianDamping;
    config.Search.Method = optlib::ParseLineSearchMethod(lineSearch);
    config.Search.InitialStep = lineSearchInitialStep;
    config.Search.Reduction = lineSearchReduction;
    config.StoreTrajectory = logTrajectory;
    return config;
}

optlib::ZeroOrderConfig MakeZeroOrderConfig(std::size_t maxIter,
                                            double stepTolerance,
                                            double functionTolerance,
                                            double initialStep,
                                            double lineSearchRadius,
                                            double lineSearchTolerance,
                                            bool logTrajectory)
{
    optlib::ZeroOrderConfig config;
    config.Criteria.MaxIterations = maxIter;
    config.Criteria.StepTolerance = stepTolerance;
    config.Criteria.FunctionTolerance = functionTolerance;
    config.InitialStep = initialStep;
    config.LineSearchRadius = lineSearchRadius;
    config.LineSearchTolerance = lineSearchTolerance;
    config.StoreTrajectory = logTrajectory;
    return config;
}

double LearningRateAtBinding(std::size_t iteration,
                             const std::string& schedule,
                             double initialLearningRate,
                             double gamma,
                             std::size_t stepSize,
                             double decayRate,
                             double minimumLearningRate,
                             std::size_t totalIterations,
                             std::size_t warmupSteps)
{
    optlib::LearningRateConfig config;
    config.Schedule = optlib::ParseLearningRateSchedule(schedule);
    config.InitialLearningRate = initialLearningRate;
    config.Gamma = gamma;
    config.StepSize = stepSize;
    config.DecayRate = decayRate;
    config.MinimumLearningRate = minimumLearningRate;
    config.TotalIterations = totalIterations;
    config.WarmupSteps = warmupSteps;
    return optlib::LearningRateAt(iteration, config);
}

py::dict TrajectoryToDict(const optlib::Trajectory& path)
{
    py::dict output;
    output["x"] = SpanToMatrixArray(path.Size(), path.Dimension(), path.Points());
    output["f"] = SpanToVectorArray(path.Values());
    output["grad_norm"] = SpanToVectorArray(path.GradientNorms());
    output["time_ms"] = SpanToVectorArray(path.TimesMs());
    return output;
}

py::dict OptimizeResultToDict(optlib::OptimizeResult result)
{
    py::dict output;
    output["x"] = VectorToArray(std::move(result.X));
    output["value"] = result.Value;
    output["gradient_norm"] = result.GradientNorm;
    output["iterations"] = result.Iterations;
    output["function_evaluations"] = result.FunctionEvaluations;
    output["gradient_evaluations"] = result.GradientEvaluations;
    output["hessian_evaluations"] = result.HessianEvaluations;
    output["converged"] = result.Converged;
    output["trajectory"] = TrajectoryToDict(result.Path);
    return output;
}

py::array_t<double> PointArray(std::span<const double> values)
{
    py::array_t<double> point(static_cast<py::ssize_t>(values.size()));
    auto mutablePoint = point.mutable_unchecked<1>();
    for (py::ssize_t index = 0; index < mutablePoint.shape(0); ++index) {
        mutablePoint(index) = values[static_cast<std::size_t>(index)];
    }
    return point;
}

py::array_t<double> NumericGradient(const py::function& function,
                                    const Array& x,
                                    const std::string& scheme,
                                    double step)
{
    auto xValue = VectorFromArray(x);
    auto options = optlib::DifferentiationOptions{.Scheme = ParseScheme(scheme), .Step = step};
    auto wrappedFunction = [&function](std::span<const double> values) {
        py::gil_scoped_acquire acquire;
        return function(PointArray(values)).cast<double>();
    };

    optlib::Vector gradient;
    {
        py::gil_scoped_release release;
        gradient = optlib::ComputeGradient(wrappedFunction, xValue.Span(), options);
    }
    return VectorToArray(std::move(gradient));
}

py::array_t<double> RosenbrockNumericalGradient(const Array& x,
                                                const std::string& scheme,
                                                double step)
{
    auto xValue = VectorFromArray(x);
    auto options = optlib::DifferentiationOptions{.Scheme = ParseScheme(scheme), .Step = step};
    auto function = [](std::span<const double> values) {
        return optlib::RosenbrockValue(values);
    };
    optlib::Vector gradient;
    {
        py::gil_scoped_release release;
        gradient = optlib::ComputeGradient(function, xValue.Span(), options);
    }
    return VectorToArray(std::move(gradient));
}

py::array_t<double> RosenbrockAutogradGradient(const Array& x)
{
    auto xValue = VectorFromArray(x);
    auto function = [](std::span<const optlib::Dual> values) {
        return optlib::Rosenbrock(values.size()).Eval<optlib::Dual>(values);
    };
    optlib::Vector gradient;
    {
        py::gil_scoped_release release;
        gradient = optlib::ComputeAutogradGradient(function, xValue.Span());
    }
    return VectorToArray(std::move(gradient));
}

py::dict MinimizeRosenbrockBinding(const Array& x0,
                                   const std::string& method,
                                   double learningRate,
                                   std::size_t maxIter,
                                   double gradientTolerance,
                                   double stepTolerance,
                                   double functionTolerance,
                                   double momentum,
                                   double beta1,
                                   double beta2,
                                   double epsilon,
                                   bool logTrajectory,
                                   const std::string& schedule,
                                   double learningRateGamma,
                                   std::size_t learningRateStepSize,
                                   double learningRateDecay,
                                   double minimumLearningRate,
                                   std::size_t warmupSteps,
                                   std::size_t scheduleIterations)
{
    auto xValue = VectorFromArray(x0);
    auto config = MakeFirstOrderConfig(learningRate,
                                       maxIter,
                                       gradientTolerance,
                                       stepTolerance,
                                       functionTolerance,
                                       momentum,
                                       beta1,
                                       beta2,
                                       epsilon,
                                       logTrajectory,
                                       schedule,
                                       learningRateGamma,
                                       learningRateStepSize,
                                       learningRateDecay,
                                       minimumLearningRate,
                                       warmupSteps,
                                       scheduleIterations);
    optlib::OptimizeResult result;
    {
        py::gil_scoped_release release;
        result = optlib::MinimizeRosenbrock(
            xValue.Span(), optlib::ParseFirstOrderMethod(method), config);
    }
    return OptimizeResultToDict(std::move(result));
}

py::dict MinimizeBinding(const py::function& valueFunction,
                         const py::function& gradientFunction,
                         const Array& x0,
                         const std::string& method,
                         double learningRate,
                         std::size_t maxIter,
                         double gradientTolerance,
                         double stepTolerance,
                         double functionTolerance,
                         double momentum,
                         double beta1,
                         double beta2,
                         double epsilon,
                         bool logTrajectory,
                         const std::string& schedule,
                         double learningRateGamma,
                         std::size_t learningRateStepSize,
                         double learningRateDecay,
                         double minimumLearningRate,
                         std::size_t warmupSteps,
                         std::size_t scheduleIterations)
{
    auto xValue = VectorFromArray(x0);
    auto config = MakeFirstOrderConfig(learningRate,
                                       maxIter,
                                       gradientTolerance,
                                       stepTolerance,
                                       functionTolerance,
                                       momentum,
                                       beta1,
                                       beta2,
                                       epsilon,
                                       logTrajectory,
                                       schedule,
                                       learningRateGamma,
                                       learningRateStepSize,
                                       learningRateDecay,
                                       minimumLearningRate,
                                       warmupSteps,
                                       scheduleIterations);
    auto wrappedValueFunction = [&valueFunction](std::span<const double> values) {
        py::gil_scoped_acquire acquire;
        return valueFunction(PointArray(values)).cast<double>();
    };
    auto wrappedGradientFunction = [&gradientFunction](std::span<const double> values,
                                                       std::span<double> gradient) {
        py::gil_scoped_acquire acquire;
        auto gradientArray = gradientFunction(PointArray(values)).cast<Array>();
        auto gradientSpan = VectorSpan(gradientArray);
        if (gradientSpan.size() != gradient.size()) {
            throw std::invalid_argument("Python gradient returned an unexpected shape");
        }
        optlib::Copy(gradientSpan, gradient);
    };

    optlib::OptimizeResult result;
    {
        py::gil_scoped_release release;
        result = optlib::MinimizeFirstOrder(wrappedValueFunction,
                                            wrappedGradientFunction,
                                            xValue.Span(),
                                            optlib::ParseFirstOrderMethod(method),
                                            config);
    }
    return OptimizeResultToDict(std::move(result));
}

py::dict MinimizeRosenbrockSecondOrderBinding(const Array& x0,
                                              const std::string& method,
                                              std::size_t maxIter,
                                              double gradientTolerance,
                                              double stepTolerance,
                                              double functionTolerance,
                                              std::size_t historySize,
                                              double hessianDamping,
                                              const std::string& lineSearch,
                                              double lineSearchInitialStep,
                                              double lineSearchReduction,
                                              bool logTrajectory)
{
    auto xValue = VectorFromArray(x0);
    auto config = MakeSecondOrderConfig(maxIter,
                                        gradientTolerance,
                                        stepTolerance,
                                        functionTolerance,
                                        historySize,
                                        hessianDamping,
                                        lineSearch,
                                        lineSearchInitialStep,
                                        lineSearchReduction,
                                        logTrajectory);
    optlib::OptimizeResult result;
    {
        py::gil_scoped_release release;
        result = optlib::MinimizeRosenbrockSecondOrder(
            xValue.Span(), optlib::ParseSecondOrderMethod(method), config);
    }
    return OptimizeResultToDict(std::move(result));
}

py::dict MinimizeSecondOrderBinding(const py::function& valueFunction,
                                    const py::function& gradientFunction,
                                    const Array& x0,
                                    const std::string& method,
                                    std::size_t maxIter,
                                    double gradientTolerance,
                                    double stepTolerance,
                                    double functionTolerance,
                                    std::size_t historySize,
                                    double hessianDamping,
                                    const std::string& lineSearch,
                                    double lineSearchInitialStep,
                                    double lineSearchReduction,
                                    const py::object& hessianFunction,
                                    bool logTrajectory)
{
    auto xValue = VectorFromArray(x0);
    auto config = MakeSecondOrderConfig(maxIter,
                                        gradientTolerance,
                                        stepTolerance,
                                        functionTolerance,
                                        historySize,
                                        hessianDamping,
                                        lineSearch,
                                        lineSearchInitialStep,
                                        lineSearchReduction,
                                        logTrajectory);
    auto wrappedValueFunction = [&valueFunction](std::span<const double> values) {
        py::gil_scoped_acquire acquire;
        return valueFunction(PointArray(values)).cast<double>();
    };
    auto wrappedGradientFunction = [&gradientFunction](std::span<const double> values,
                                                       std::span<double> gradient) {
        py::gil_scoped_acquire acquire;
        auto gradientArray = gradientFunction(PointArray(values)).cast<Array>();
        auto gradientSpan = VectorSpan(gradientArray);
        if (gradientSpan.size() != gradient.size()) {
            throw std::invalid_argument("Python gradient returned an unexpected shape");
        }
        optlib::Copy(gradientSpan, gradient);
    };

    optlib::HessianFunction wrappedHessianFunction;
    if (!hessianFunction.is_none()) {
        py::function hessianCallable = hessianFunction.cast<py::function>();
        wrappedHessianFunction = [hessianCallable](std::span<const double> values,
                                                   optlib::Matrix& hessian) {
            py::gil_scoped_acquire acquire;
            auto hessianArray = hessianCallable(PointArray(values)).cast<Array>();
            hessian = MatrixFromArray(hessianArray);
        };
    }

    auto parsedMethod = optlib::ParseSecondOrderMethod(method);
    if (parsedMethod == optlib::SecondOrderMethod::Newton && !wrappedHessianFunction) {
        throw std::invalid_argument("Newton requires a Hessian callable");
    }

    optlib::OptimizeResult result;
    {
        py::gil_scoped_release release;
        result = optlib::MinimizeSecondOrder(wrappedValueFunction,
                                             wrappedGradientFunction,
                                             wrappedHessianFunction,
                                             xValue.Span(),
                                             parsedMethod,
                                             config);
    }
    return OptimizeResultToDict(std::move(result));
}

py::dict MinimizeRosenbrockZeroOrderBinding(const Array& x0,
                                            const std::string& method,
                                            std::size_t maxIter,
                                            double stepTolerance,
                                            double functionTolerance,
                                            double initialStep,
                                            double lineSearchRadius,
                                            double lineSearchTolerance,
                                            bool logTrajectory)
{
    auto xValue = VectorFromArray(x0);
    auto config = MakeZeroOrderConfig(maxIter,
                                      stepTolerance,
                                      functionTolerance,
                                      initialStep,
                                      lineSearchRadius,
                                      lineSearchTolerance,
                                      logTrajectory);
    optlib::OptimizeResult result;
    {
        py::gil_scoped_release release;
        result = optlib::MinimizeRosenbrockZeroOrder(
            xValue.Span(), optlib::ParseZeroOrderMethod(method), config);
    }
    return OptimizeResultToDict(std::move(result));
}

py::dict MinimizeZeroOrderBinding(const py::function& valueFunction,
                                  const Array& x0,
                                  const std::string& method,
                                  std::size_t maxIter,
                                  double stepTolerance,
                                  double functionTolerance,
                                  double initialStep,
                                  double lineSearchRadius,
                                  double lineSearchTolerance,
                                  bool logTrajectory)
{
    auto xValue = VectorFromArray(x0);
    auto config = MakeZeroOrderConfig(maxIter,
                                      stepTolerance,
                                      functionTolerance,
                                      initialStep,
                                      lineSearchRadius,
                                      lineSearchTolerance,
                                      logTrajectory);
    auto wrappedValueFunction = [&valueFunction](std::span<const double> values) {
        py::gil_scoped_acquire acquire;
        return valueFunction(PointArray(values)).cast<double>();
    };
    optlib::OptimizeResult result;
    {
        py::gil_scoped_release release;
        result = optlib::MinimizeZeroOrder(
            wrappedValueFunction, xValue.Span(), optlib::ParseZeroOrderMethod(method), config);
    }
    return OptimizeResultToDict(std::move(result));
}

} // namespace

PYBIND11_MODULE(_optlib, moduleHandle)
{
    moduleHandle.doc() = "C++23 optimization core for optlib.";
    moduleHandle.attr("__version__") = std::string(optlib::Version());

    py::enum_<optlib::DifferentiationScheme>(moduleHandle, "DifferentiationScheme")
        .value("Forward", optlib::DifferentiationScheme::Forward)
        .value("Central", optlib::DifferentiationScheme::Central)
        .value("FivePoint", optlib::DifferentiationScheme::FivePoint)
        .export_values();
    py::enum_<optlib::FirstOrderMethod>(moduleHandle, "FirstOrderMethod")
        .value("GradientDescent", optlib::FirstOrderMethod::GradientDescent)
        .value("HeavyBall", optlib::FirstOrderMethod::HeavyBall)
        .value("Nesterov", optlib::FirstOrderMethod::Nesterov)
        .value("Adam", optlib::FirstOrderMethod::Adam)
        .value("RMSProp", optlib::FirstOrderMethod::RMSProp)
        .value("Adagrad", optlib::FirstOrderMethod::Adagrad)
        .export_values();
    py::enum_<optlib::LearningRateSchedule>(moduleHandle, "LearningRateSchedule")
        .value("Constant", optlib::LearningRateSchedule::Constant)
        .value("Step", optlib::LearningRateSchedule::Step)
        .value("Exponential", optlib::LearningRateSchedule::Exponential)
        .value("Cosine", optlib::LearningRateSchedule::Cosine)
        .export_values();
    py::enum_<optlib::LineSearchMethod>(moduleHandle, "LineSearchMethod")
        .value("Armijo", optlib::LineSearchMethod::Armijo)
        .value("StrongWolfe", optlib::LineSearchMethod::StrongWolfe)
        .export_values();
    py::enum_<optlib::SecondOrderMethod>(moduleHandle, "SecondOrderMethod")
        .value("Newton", optlib::SecondOrderMethod::Newton)
        .value("BFGS", optlib::SecondOrderMethod::BFGS)
        .value("LBFGS", optlib::SecondOrderMethod::LBFGS)
        .export_values();
    py::enum_<optlib::ZeroOrderMethod>(moduleHandle, "ZeroOrderMethod")
        .value("NelderMead", optlib::ZeroOrderMethod::NelderMead)
        .value("Powell", optlib::ZeroOrderMethod::Powell)
        .value("CoordinateSearch", optlib::ZeroOrderMethod::CoordinateSearch)
        .export_values();

    moduleHandle.def("Version", []() { return std::string(optlib::Version()); });
    moduleHandle.def("Add", &optlib::Add, py::arg("left_value"), py::arg("right_value"));
    moduleHandle.def("LearningRateAt", &LearningRateAtBinding, py::arg("iteration"),
                     py::arg("schedule") = "constant", py::arg("initial_learning_rate") = 1e-3,
                     py::arg("gamma") = 0.5, py::arg("step_size") = 100,
                     py::arg("decay_rate") = 1e-3,
                     py::arg("minimum_learning_rate") = 0.0,
                     py::arg("total_iterations") = 1000, py::arg("warmup_steps") = 0);
    moduleHandle.def(
        "Dot",
        [](const Array& left, const Array& right) {
            auto leftSpan = VectorSpan(left);
            auto rightSpan = VectorSpan(right);
            py::gil_scoped_release release;
            return optlib::Dot(leftSpan, rightSpan);
        },
        py::arg("left"),
        py::arg("right"));
    moduleHandle.def(
        "Norm2",
        [](const Array& values) {
            auto valueSpan = VectorSpan(values);
            py::gil_scoped_release release;
            return optlib::Norm2(valueSpan);
        },
        py::arg("values"));
    moduleHandle.def(
        "NormInf",
        [](const Array& values) {
            auto valueSpan = VectorSpan(values);
            py::gil_scoped_release release;
            return optlib::NormInf(valueSpan);
        },
        py::arg("values"));
    moduleHandle.def(
        "Axpy",
        [](double alpha, const Array& x, const Array& y) {
            auto result = VectorFromArray(y);
            auto xSpan = VectorSpan(x);
            {
                py::gil_scoped_release release;
                optlib::Axpy(alpha, xSpan, result.Span());
            }
            return VectorToArray(std::move(result));
        },
        py::arg("alpha"),
        py::arg("x"),
        py::arg("y"));
    moduleHandle.def(
        "Gemv",
        [](const Array& matrix, const Array& vector) {
            auto matrixValue = MatrixFromArray(matrix);
            auto vectorValue = VectorFromArray(vector);
            optlib::Vector result;
            {
                py::gil_scoped_release release;
                result = optlib::Gemv(matrixValue, vectorValue);
            }
            return VectorToArray(std::move(result));
        },
        py::arg("matrix"),
        py::arg("vector"));
    moduleHandle.def(
        "Gemm",
        [](const Array& left, const Array& right) {
            auto leftValue = MatrixFromArray(left);
            auto rightValue = MatrixFromArray(right);
            optlib::Matrix result;
            {
                py::gil_scoped_release release;
                result = optlib::Gemm(leftValue, rightValue);
            }
            return MatrixToArray(std::move(result));
        },
        py::arg("left"),
        py::arg("right"));
    moduleHandle.def(
        "RosenbrockValue",
        [](const Array& x) {
            auto xSpan = VectorSpan(x);
            py::gil_scoped_release release;
            return optlib::RosenbrockValue(xSpan);
        },
        py::arg("x"));
    moduleHandle.def(
        "RosenbrockGradient",
        [](const Array& x) {
            auto xSpan = VectorSpan(x);
            optlib::Vector gradient;
            {
                py::gil_scoped_release release;
                gradient = optlib::RosenbrockGradient(xSpan);
            }
            return VectorToArray(std::move(gradient));
        },
        py::arg("x"));
    moduleHandle.def(
        "RosenbrockHessian",
        [](const Array& x) {
            auto xSpan = VectorSpan(x);
            optlib::Matrix hessian;
            {
                py::gil_scoped_release release;
                hessian = optlib::RosenbrockHessian(xSpan);
            }
            return MatrixToArray(std::move(hessian));
        },
        py::arg("x"));
    moduleHandle.def("RosenbrockNumericalGradient", &RosenbrockNumericalGradient, py::arg("x"),
                     py::arg("scheme") = "central", py::arg("step") = 0.0);
    moduleHandle.def("RosenbrockGradientNumeric", &RosenbrockNumericalGradient, py::arg("x"),
                     py::arg("scheme") = "central", py::arg("step") = 0.0);
    moduleHandle.def("RosenbrockAutogradGradient", &RosenbrockAutogradGradient, py::arg("x"));
    moduleHandle.def("RosenbrockGradientAutograd", &RosenbrockAutogradGradient, py::arg("x"));
    moduleHandle.def("NumericGradient", &NumericGradient, py::arg("function"), py::arg("x"),
                     py::arg("scheme") = "central", py::arg("step") = 0.0);
    moduleHandle.def("MinimizeRosenbrock", &MinimizeRosenbrockBinding, py::arg("x0"),
                     py::arg("method") = "adam", py::arg("learning_rate") = 1e-3,
                     py::arg("max_iter") = 10000, py::arg("gradient_tolerance") = 1e-6,
                     py::arg("step_tolerance") = 1e-12,
                     py::arg("function_tolerance") = 1e-12, py::arg("momentum") = 0.9,
                     py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
                     py::arg("epsilon") = 1e-8, py::arg("log_trajectory") = true,
                     py::arg("schedule") = "constant", py::arg("learning_rate_gamma") = 0.5,
                     py::arg("learning_rate_step_size") = 100,
                     py::arg("learning_rate_decay") = 1e-3,
                     py::arg("minimum_learning_rate") = 0.0, py::arg("warmup_steps") = 0,
                     py::arg("schedule_iterations") = 0);
    moduleHandle.def("Minimize", &MinimizeBinding, py::arg("value_function"),
                     py::arg("gradient_function"), py::arg("x0"), py::arg("method") = "adam",
                     py::arg("learning_rate") = 1e-3, py::arg("max_iter") = 10000,
                     py::arg("gradient_tolerance") = 1e-6,
                     py::arg("step_tolerance") = 1e-12,
                     py::arg("function_tolerance") = 1e-12, py::arg("momentum") = 0.9,
                     py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
                     py::arg("epsilon") = 1e-8, py::arg("log_trajectory") = true,
                     py::arg("schedule") = "constant", py::arg("learning_rate_gamma") = 0.5,
                     py::arg("learning_rate_step_size") = 100,
                     py::arg("learning_rate_decay") = 1e-3,
                     py::arg("minimum_learning_rate") = 0.0, py::arg("warmup_steps") = 0,
                     py::arg("schedule_iterations") = 0);
    moduleHandle.def("MinimizeRosenbrockSecondOrder", &MinimizeRosenbrockSecondOrderBinding,
                     py::arg("x0"), py::arg("method") = "bfgs", py::arg("max_iter") = 1000,
                     py::arg("gradient_tolerance") = 1e-6,
                     py::arg("step_tolerance") = 1e-12,
                     py::arg("function_tolerance") = 1e-12, py::arg("history_size") = 10,
                     py::arg("hessian_damping") = 1e-6,
                     py::arg("line_search") = "strong_wolfe",
                     py::arg("line_search_initial_step") = 1.0,
                     py::arg("line_search_reduction") = 0.5,
                     py::arg("log_trajectory") = true);
    moduleHandle.def("MinimizeSecondOrder", &MinimizeSecondOrderBinding,
                     py::arg("value_function"), py::arg("gradient_function"),
                     py::arg("x0"), py::arg("method") = "lbfgs",
                     py::arg("max_iter") = 1000,
                     py::arg("gradient_tolerance") = 1e-6,
                     py::arg("step_tolerance") = 1e-12,
                     py::arg("function_tolerance") = 1e-12, py::arg("history_size") = 10,
                     py::arg("hessian_damping") = 1e-6,
                     py::arg("line_search") = "strong_wolfe",
                     py::arg("line_search_initial_step") = 1.0,
                     py::arg("line_search_reduction") = 0.5,
                     py::arg("hessian_function") = py::none(),
                     py::arg("log_trajectory") = true);
    moduleHandle.def("MinimizeRosenbrockZeroOrder", &MinimizeRosenbrockZeroOrderBinding,
                     py::arg("x0"), py::arg("method") = "nelder_mead",
                     py::arg("max_iter") = 2000, py::arg("step_tolerance") = 1e-8,
                     py::arg("function_tolerance") = 1e-10, py::arg("initial_step") = 1.0,
                     py::arg("line_search_radius") = 1.0,
                     py::arg("line_search_tolerance") = 1e-6,
                     py::arg("log_trajectory") = true);
    moduleHandle.def("MinimizeZeroOrder", &MinimizeZeroOrderBinding, py::arg("value_function"),
                     py::arg("x0"), py::arg("method") = "nelder_mead",
                     py::arg("max_iter") = 2000, py::arg("step_tolerance") = 1e-8,
                     py::arg("function_tolerance") = 1e-10, py::arg("initial_step") = 1.0,
                     py::arg("line_search_radius") = 1.0,
                     py::arg("line_search_tolerance") = 1e-6,
                     py::arg("log_trajectory") = true);
}
