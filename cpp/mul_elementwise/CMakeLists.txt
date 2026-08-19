cmake_minimum_required(VERSION 3.16)

project(examples-minimal-dependencies)

add_compile_options(-Werror)

# find dlpack
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")
include(CheckDlpack)

find_library(TOOLKIT qant_native_computing_toolkit REQUIRED)

add_executable(pure_cpp pure_cpp.cpp)
target_link_libraries(pure_cpp PRIVATE ${TOOLKIT} dlpack::dlpack)
# required for the bfloat type:
target_compile_features(pure_cpp PUBLIC cxx_std_23)
