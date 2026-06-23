#include <pybind11/pybind11.h>

#include <optlib/core/Version.hpp>

#include <string>

namespace py = pybind11;

PYBIND11_MODULE(_optlib, moduleHandle)
{
    moduleHandle.doc() = "C++23 optimization core for optlib.";
    moduleHandle.attr("__version__") = std::string(optlib::Version());

    moduleHandle.def("Version", []() { return std::string(optlib::Version()); },
                     "Return the optlib package version.");
    moduleHandle.def("Add", &optlib::Add, py::arg("left_value"), py::arg("right_value"),
                     "Return the sum of two floating-point values.");
}
