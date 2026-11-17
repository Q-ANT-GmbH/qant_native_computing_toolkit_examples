cmake_minimum_required(VERSION 3.16)

project(dft-npu)

add_compile_options(-Werror)

# find dlpack
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")
include(CheckDlpack)

find_library(TOOLKIT qant_native_computing_toolkit REQUIRED)

add_executable(dft_npu dft_npu.cpp)
target_link_libraries(dft_npu PRIVATE ${TOOLKIT} dlpack::dlpack)
# required for the bfloat type:
target_compile_features(dft_npu PUBLIC cxx_std_23)
