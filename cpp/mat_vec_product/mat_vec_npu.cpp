/**
 * This file is an executable to demonstrate the basic usage of Q.ANT native computing toolkit.
 * We use xtensor as the cpp tensor backend. This is not a requirement.
 * Any tensor/data storage framework (eigen3, std::vector, ...) can be used,
 * as long as it can be converted to a DLPack tensor.
 */

#include <iostream>
#include <stdfloat> // for bfloat16_t (C++23)
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>

#include "dlpack_utils.h"
#include "qant_native_computing_toolkit.h"

using namespace qant_native_computing_toolkit;

// Performs a matrix-vector multiplication and displays the result.
int main()
{
    auto n_rows = 10;
    auto n_cols = 5;

    auto npu_id = 0;

    // We will use the forward pass through a linear neural network layer to do the matrix-vector product.
    // The input tensors need additional dimensions, but we will only use the first two.
    xt::xarray<float> vec = xt::random::rand({1, n_cols}, -1., 1.);
    xt::xarray<float> mat = xt::random::rand({n_rows, n_cols}, -1., 1.);

    // convert to bfloat16.
    xt::xarray<std::bfloat16_t> vec_bf16 = xt::cast<std::bfloat16_t>(vec);
    xt::xarray<std::bfloat16_t> mat_bf16 = xt::cast<std::bfloat16_t>(mat);

    // convert xtensor arrays to DLPack tensors.
    auto vec_mtensor = xarray_to_mtensor_view(vec_bf16);
    auto mat_mtensor = xarray_to_mtensor_view(mat_bf16);

    // initialise the NPU and perform the operation.
    auto res_qant_mtensor = native::linear_fprop(npu_id, &vec_mtensor, &mat_mtensor);
    if (res_qant_mtensor == NULL)
    {
        exit(-1);
    }
    // here we could add more operations and build a neural network.

    // convert DLPack tensor to xtensor array.
    auto res_qant_bf16 = mtensor_to_tensor(res_qant_mtensor, false);

    // convert bfloat16 to float.
    xt::xarray<float> res_qant = xt::cast<float>(res_qant_bf16);

    // for comparison: calculate product on CPU
    auto res_cpu = xt::linalg::tensordot(mat, vec, {1}, {1});

    // for printing: remove extra dimensions
    std::cout << "result NPU" << std::endl
              << res_qant.reshape({-1}) << std::endl;
    std::cout << "result CPU" << std::endl
              << res_cpu.reshape({-1}) << std::endl;

    // Optional step: release the NPU to free memory.
    generic::release_npu(npu_id);

    // DLPack requires manual memory management => call destructor
    // You could implement this as the actual destructor of the DLManagedTensor
    // class if you want to modify dlpack.h
    vec_mtensor.deleter(&vec_mtensor);
    mat_mtensor.deleter(&mat_mtensor);
    res_qant_mtensor->deleter(res_qant_mtensor);

    return 0;
}
