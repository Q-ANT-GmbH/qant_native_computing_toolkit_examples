# make sure we find and import the dlpack header of the submodule and not any other, system-wide version
get_filename_component(_cmake_dir "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(_cpp_root "${_cmake_dir}/.." ABSOLUTE)

set(_dlpack_include "${_cpp_root}/../dlpack/include")

if(NOT EXISTS "${_dlpack_include}/dlpack/dlpack.h")
    message(FATAL_ERROR "dlpack submodule missing - run: git submodule update --init")
endif()

if(NOT TARGET dlpack::dlpack)
    add_library(dlpack::dlpack INTERFACE IMPORTED)
    target_include_directories(dlpack::dlpack INTERFACE "${_dlpack_include}")
endif()