/**
 * This file is a standalone executable, serving as an example of cpp usage with minimal dependencies.
 */

#include <dlpack/dlpack.h>
#include <iostream>
#include <vector>

#include "qant_native_computing_toolkit.h"

using namespace qant_native_computing_toolkit;

void print_array_float(const std::vector<float> &vec)
{
    for (auto elem : vec)
    {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}

void delete_mtensor(struct DLManagedTensorVersioned *self)
{
    delete[] self->dl_tensor.shape;
}

/// @brief Utility function to create a DLManagedTensorVersioned based on a std::vector
/// @param vec The vector that holds the data
/// @return The DLManagedTensorVersioned that wraps the data owned by the input vector.
DLManagedTensorVersioned vector_to_mtensor(const std::vector<float> &vec)
{
    int64_t const len = vec.size();

    DLTensor tensor = DLTensor();
    tensor.ndim = 1;
    tensor.shape = new int64_t[tensor.ndim];
    tensor.shape[0] = len;
    tensor.strides = NULL;

    tensor.dtype.code = kDLFloat;
    tensor.dtype.bits = 32;
    tensor.dtype.lanes = 1;

    tensor.data = const_cast<void *>(static_cast<const void *>(vec.data()));
    tensor.byte_offset = 0;
    tensor.device.device_type = kDLCPU;
    tensor.device.device_id = 0;

    DLManagedTensorVersioned mtensor = DLManagedTensorVersioned();
    mtensor.version = DLPackVersion{major : DLPACK_MAJOR_VERSION, minor : DLPACK_MINOR_VERSION};
    mtensor.dl_tensor = tensor;
    mtensor.manager_ctx = const_cast<void *>(static_cast<const void *>(&vec));
    mtensor.deleter = delete_mtensor;

    return mtensor;
}

/// @brief Util function to extract a vector from a DLManagedTensorVersioned
/// @param tensor The DLManagedTensorVersioned that holds the data
/// @return A vector containing the data that the DLManagedTensorVersioned owns
std::vector<float> mtensor_to_vector(DLManagedTensorVersioned const *tensor)
{
    float *data = static_cast<float *>(tensor->dl_tensor.data);
    std::vector<float> vec(data, data + tensor->dl_tensor.shape[0]);
    return vec;
}

int main()
{
    int n_elems = 10;
    int npu_id = 0;

    std::vector<float> us = {0.9, 0.8, 0.7, 0.6};
    std::vector<float> vs = {0.5, 0.3, 0.2, 0.1};
    auto us_tensor = vector_to_mtensor(us);
    auto vs_tensor = vector_to_mtensor(vs);

    auto res_npu_mtensor = native::mul_npu_f32(npu_id, &us_tensor, &vs_tensor);
    if (res_npu_mtensor == nullptr)
    {
        exit(-1);
    }

    auto res_npu_vec = mtensor_to_vector(res_npu_mtensor);

    std::vector<float> res_cpu_vec(us.size());

    for (size_t i = 0; i < res_cpu_vec.size(); ++i)
    {
        res_cpu_vec[i] = us[i] * vs[i];
    }

    std::cout << "us" << std::endl;
    print_array_float(us);
    std::cout << "vs" << std::endl;
    print_array_float(vs);
    std::cout << "multiplied on npu" << std::endl;
    print_array_float(res_npu_vec);
    std::cout << "multiplied on cpu" << std::endl;
    print_array_float(res_cpu_vec);

    generic::release_npu(npu_id);
    us_tensor.deleter(&us_tensor);
    vs_tensor.deleter(&vs_tensor);
    res_npu_mtensor->deleter(res_npu_mtensor);

    return 0;
}
