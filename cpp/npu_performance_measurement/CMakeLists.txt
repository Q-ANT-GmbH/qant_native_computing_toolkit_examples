cmake_minimum_required(VERSION 3.16)

project(npu-performance-measurement)

add_compile_options(-Werror)

# find dlpack
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")
include(CheckDlpack)

find_library(TOOLKIT qant_native_computing_toolkit REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE ${TOOLKIT} dlpack::dlpack)
target_compile_features(main PUBLIC cxx_std_23)
