cmake_minimum_required(VERSION 3.16)

project(examples-minimal-dependencies)

add_compile_options(-Werror)

get_filename_component(CONDA_PREFIX_ENV "$ENV{CONDA_PREFIX}" ABSOLUTE)
include_directories(${CONDA_PREFIX_ENV}/include)

# find dlpack
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")
include(CheckDlpack)

find_library(TOOLKIT qant_native_computing_toolkit REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE ${TOOLKIT} dlpack::dlpack)
target_compile_features(main PUBLIC cxx_std_23)
