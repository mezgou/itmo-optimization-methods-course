#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <optlib/core/Differentiation.hpp>
#include <optlib/core/LinAlg.hpp>
#include <optlib/core/Version.hpp>
#include <optlib/core/functions/Rosenbrock.hpp>

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

py::array_t<double> NumericGradient(const py::function& function,
                                    const Array& x,
                                    const std::string& scheme,
                                    double step)
{
    auto xValue = VectorFromArray(x);
    auto options = optlib::DifferentiationOptions{.Scheme = ParseScheme(scheme), .Step = step};
    auto wrappedFunction = [&function](std::span<const double> values) {
        py::gil_scoped_acquire acquire;
        py::array_t<double> point(static_cast<py::ssize_t>(values.size()));
        auto mutablePoint = point.mutable_unchecked<1>();
        for (py::ssize_t index = 0; index < mutablePoint.shape(0); ++index) {
            mutablePoint(index) = values[static_cast<std::size_t>(index)];
        }
        return function(point).cast<double>();
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

    moduleHandle.def("Version", []() { return std::string(optlib::Version()); });
    moduleHandle.def("Add", &optlib::Add, py::arg("left_value"), py::arg("right_value"));
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
}
