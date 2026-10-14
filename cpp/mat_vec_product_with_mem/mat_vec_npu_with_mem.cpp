/**
 * This file is an executable to demonstrate the explicit memory-management API of the
 * Q.ANT native computing toolkit. Unlike ../mat_vec_product (which uses the one-shot
 * native::linear_fprop() call), here the weights and features are explicitly allocated and
 * uploaded to NPU memory, the product is executed directly on that resident data, and only
 * the result is downloaded. This lets the caller keep e.g. a set of weights resident on the
 * NPU across many calls, uploading only the (much smaller) features each time.
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

// Performs a linear forward propagation via the explicit memory-management API and displays the result.
int main()
{
    auto n_channels_out = 10;
    auto n_channels_in = 5;

    auto npu_id = 0;

    // The input tensors need additional dimensions, but we will only use the first two.
    xt::xarray<float> features = xt::random::rand({1, n_channels_in}, -1., 1.);
    xt::xarray<float> weights = xt::random::rand({n_channels_out, n_channels_in}, -1., 1.);

    // convert to bfloat16.
    xt::xarray<std::bfloat16_t> features_bf16 = xt::cast<std::bfloat16_t>(features);
    xt::xarray<std::bfloat16_t> weights_bf16 = xt::cast<std::bfloat16_t>(weights);

    // convert xtensor arrays to DLPack tensors.
    auto features_mtensor = xarray_to_mtensor_view(features_bf16);
    auto weights_mtensor = xarray_to_mtensor_view(weights_bf16);

    // Upload the weights to NPU memory, and allocate NPU memory for the features and the result.
    auto weights_handle = generic::npu_upload(npu_id, &weights_mtensor, generic::Weights);
    auto features_handle = generic::npu_allocate(npu_id, 1, n_channels_in, generic::Features);
    auto result_handle = generic::npu_allocate(npu_id, 1, n_channels_out, generic::Features);
    if (weights_handle == NULL || features_handle == NULL || result_handle == NULL)
    {
        exit(-1);
    }

    // Upload the features into the already allocated NPU memory; a real application would
    // repeat this step for every new input, reusing both features_handle and weights_handle.
    if (generic::npu_upload_into(features_handle, &features_mtensor) != 0)
    {
        exit(-1);
    }

    // Perform the linear forward propagation directly on the data already resident on the NPU.
    if (native::linear_fprop_preloaded(features_handle, weights_handle, result_handle) != 0)
    {
        exit(-1);
    }

    // download the result from NPU memory.
    auto res_qant_mtensor = generic::npu_download(result_handle);
    if (res_qant_mtensor == NULL)
    {
        exit(-1);
    }

    // convert DLPack tensor to xtensor array.
    auto res_qant_bf16 = mtensor_to_tensor(res_qant_mtensor, false);

    // convert bfloat16 to float.
    xt::xarray<float> res_qant = xt::cast<float>(res_qant_bf16);

    // for comparison: calculate features @ weights^T on CPU
    auto res_cpu = xt::linalg::tensordot(features, weights, {1}, {1});

    // for printing: remove extra dimensions
    std::cout << "result NPU" << std::endl
              << res_qant.reshape({-1}) << std::endl;
    std::cout << "result CPU" << std::endl
              << res_cpu.reshape({-1}) << std::endl;

    // release the NPU memory handles.
    generic::npu_free(weights_handle);
    generic::npu_free(features_handle);
    generic::npu_free(result_handle);

    // Optional step: release the NPU to free memory.
    generic::release_npu(npu_id);

    // DLPack requires manual memory management => call destructor
    features_mtensor.deleter(&features_mtensor);
    weights_mtensor.deleter(&weights_mtensor);
    res_qant_mtensor->deleter(res_qant_mtensor);

    return 0;
}
