/**
 * This file is a standalone executable, serving as an example of cpp usage with minimal dependencies.
 * It also showcases hwo to access performance measurement data.
 */

#include <cstddef>
#include <cstdlib>
#include <dlpack/dlpack.h>
#include <iostream>
#include <ostream>
#include <vector>

#include "qant_native_computing_toolkit.h"

using namespace qant_native_computing_toolkit;
using namespace qant_native_computing_toolkit::info;

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
    auto npu_id = 0;
    size_t n_elems = 2 * 4096 * 4096;

    std::cout << "Run " << n_elems << " multiplications." << std::endl;
    auto us = std::vector<float>(n_elems, 0.5);
    auto vs = std::vector<float>(n_elems, 0.5);

    auto us_tensor = vector_to_mtensor(us);
    auto vs_tensor = vector_to_mtensor(vs);

    int error = info::reset_perf_counter(npu_id);
    if (error != 0)
    {
        generic::release_npu(npu_id);
        std::cout << "Could not reset performance counter!" << std::endl;
        return -1;
    }

    auto res_npu_mtensor = native::mul_npu_f32(npu_id, &us_tensor, &vs_tensor);
    if (res_npu_mtensor == NULL)
    {
        exit(-1);
    }
    QantPerformanceCounterInfo *counter_info = (QantPerformanceCounterInfo *)malloc(sizeof(QantPerformanceCounterInfo));

    error = get_perf_counter(npu_id, counter_info);

    if (error != 0)
    {
        generic::release_npu(npu_id);
        std::cout << "Could not read performance counter!" << std::endl;
        return -1;
    }

    auto res_npu_vec = mtensor_to_vector(res_npu_mtensor);

    std::cout << "Timebased Counter (in s): " << counter_info->timebased_counter << std::endl;
    // The stalling counter counts all the steps, where the system waits for data and is running an idle sequence
    std::cout << "Stalling Counter (in s): " << counter_info->stalling_counter << std::endl;

    // The pure NPU execution time is defined by
    // pure_optical_execution = timebased - stalling
    // This will also contain the calibration overhead
    double execution_time = counter_info->timebased_counter - counter_info->stalling_counter;

    double time_per_mul = execution_time / n_elems;
    double mul_speed = 1.0 / time_per_mul;

    // restrict the output to two decimal positions
    std::cout.precision(2);
    std::cout << "NPU Speed (operations/s): " << std::fixed << mul_speed << std::endl;

    // The overall utilization is defined by the ratio of pure operation time vs total operation time
    double utilization = 100 * execution_time / counter_info->timebased_counter;
    std::cout << "NPU Utilization: " << utilization << "%" << std::endl;

    // The power consumption of the sidecar (containing the laser, photonic integrated circuit and the detector)
    // is defined by the P = U * I
    // I can be measured:
    QantSensorInfo sensor_info = info::get_sensor_info(npu_id);
    auto voltage_12v = sensor_info.voltage_12v; // should be roughly 12V
    std::cout << "NPU 12V Power Level (V): " << voltage_12v << std::endl;
    auto current_12v = sensor_info.current_12v;
    std::cout << "NPU 12V Current (A): " << current_12v << std::endl;
    auto power_consumption = voltage_12v * current_12v;
    std::cout << "NPU Optical Components Power Consumption (W): " << power_consumption << std::endl;

    us_tensor.deleter(&us_tensor);
    vs_tensor.deleter(&vs_tensor);
    res_npu_mtensor->deleter(res_npu_mtensor);
    free(counter_info);

    generic::release_npu(npu_id);

    return 0;
}
