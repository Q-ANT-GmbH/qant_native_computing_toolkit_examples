cmake_minimum_required(VERSION 3.16)
project(matmul-parallel-example)

add_compile_options(-Werror)

# find xtensor
get_filename_component(CONDA_PREFIX_ENV "$ENV{CONDA_PREFIX}" ABSOLUTE)
include_directories(${CONDA_PREFIX_ENV}/include)
find_package(xtensor 0.25.0 EXACT REQUIRED)
find_package(xtensor-blas 0.21.0 EXACT REQUIRED)

# find dlpack
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")
include(CheckDlpack)

find_library(TOOLKIT qant_native_computing_toolkit REQUIRED)

add_executable(mat_vec_parallel mat_vec_parallel.cpp)
# include local dlpack utils
target_include_directories(mat_vec_parallel PRIVATE ${CMAKE_SOURCE_DIR}/../dlpack_utils/include)
target_link_libraries(mat_vec_parallel PRIVATE ${TOOLKIT} dlpack::dlpack)
# required for the bfloat type:
target_compile_features(mat_vec_parallel PUBLIC cxx_std_23)
