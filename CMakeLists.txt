cmake_minimum_required(VERSION 3.27)

project(optlib VERSION 1.0.0 LANGUAGES CXX)

option(OPTLIB_BUILD_TESTS "Build optlib C++ tests" ON)
option(OPTLIB_OPENMP "Enable OpenMP acceleration" OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

set(PYBIND11_FINDPYTHON ON)
set(PYBIND11_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory(extends/pybind11)

add_library(optlib_core INTERFACE)
target_include_directories(
    optlib_core INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                          $<INSTALL_INTERFACE:include>)
target_compile_features(optlib_core INTERFACE cxx_std_23)

pybind11_add_module(_optlib src/optlib/bindings/Module.cpp)
target_link_libraries(_optlib PRIVATE optlib_core)
target_compile_features(_optlib PRIVATE cxx_std_23)
target_compile_definitions(_optlib PRIVATE OPTLIB_VERSION="${PROJECT_VERSION}")

if(MSVC)
    target_compile_options(_optlib PRIVATE /O2 /permissive-)
else()
    include(CheckCXXCompilerFlag)
    target_compile_options(_optlib PRIVATE $<$<CONFIG:Release>:-O3>)
    check_cxx_compiler_flag("-march=native" OPTLIB_HAS_MARCH_NATIVE)
    if(OPTLIB_HAS_MARCH_NATIVE)
        target_compile_options(_optlib PRIVATE $<$<CONFIG:Release>:-march=native>)
    endif()
endif()

install(TARGETS _optlib LIBRARY DESTINATION optlib RUNTIME DESTINATION optlib
                        ARCHIVE DESTINATION optlib)

if(OPTLIB_BUILD_TESTS)
    enable_testing()

    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    add_subdirectory(extends/googletest EXCLUDE_FROM_ALL)

    add_executable(optlib_cpp_tests tests/cpp/TestSmoke.cpp)
    target_link_libraries(optlib_cpp_tests PRIVATE optlib_core GTest::gtest_main)
    target_compile_features(optlib_cpp_tests PRIVATE cxx_std_23)
    if(MSVC)
        target_compile_options(optlib_cpp_tests PRIVATE /permissive-)
    endif()

    include(GoogleTest)
    gtest_discover_tests(optlib_cpp_tests)
endif()
