/**
 * This file is a standalone executable, serving as an example of a batched discrete Fourier transform (DFT) on the NPU.
 * It only depends on DLPack and shows how complex numbers are passed to the NPU as bcomplex32 tensors.
 */

#include <cmath>
#include <complex>
#include <dlpack/dlpack.h>
#include <iostream>
#include <numbers>
#include <stdfloat>
#include <vector>

#include "qant_native_computing_toolkit.h"

using namespace qant_native_computing_toolkit;

/// @brief A complex number stored as two interleaved bfloat16 values, matching the memory layout of bcomplex32.
struct bcomplex32
{
    std::bfloat16_t real;
    std::bfloat16_t imag;
};
static_assert(sizeof(bcomplex32) == 4, "bcomplex32 must consist of exactly two bfloat16 values");

void delete_mtensor(struct DLManagedTensorVersioned *self)
{
    delete[] self->dl_tensor.shape;
}

/// @brief Utility function to create a 2D bcomplex32 DLManagedTensorVersioned based on a std::vector
/// @param vec The vector that holds the data in row-major order, length n_dfts * dft_size
/// @param n_dfts The number of independent DFTs (rows)
/// @param dft_size The number of complex values per DFT (columns)
/// @return The DLManagedTensorVersioned that wraps the data owned by the input vector.
DLManagedTensorVersioned vector_to_bcomplex_mtensor(const std::vector<bcomplex32> &vec, int64_t const n_dfts, int64_t const dft_size)
{
    DLTensor tensor = DLTensor();
    tensor.ndim = 2;
    tensor.shape = new int64_t[tensor.ndim];
    tensor.shape[0] = n_dfts;
    tensor.shape[1] = dft_size;
    tensor.strides = NULL;

    // one complex number made of two bfloat16 values (real, imaginary), requires DLPack >= 1.4
    tensor.dtype.code = kDLBcomplex;
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

/// @brief Util function to extract the complex values from a bcomplex32 DLManagedTensorVersioned
/// @param tensor The DLManagedTensorVersioned that holds the data
/// @return A vector containing the data that the DLManagedTensorVersioned owns, in row-major order
std::vector<std::complex<double>> bcomplex_mtensor_to_vector(DLManagedTensorVersioned const *tensor)
{
    int64_t len = 1;
    for (int32_t i = 0; i < tensor->dl_tensor.ndim; ++i)
    {
        len *= tensor->dl_tensor.shape[i];
    }

    auto *data = static_cast<bcomplex32 const *>(tensor->dl_tensor.data);
    std::vector<std::complex<double>> vec(len);
    for (int64_t i = 0; i < len; ++i)
    {
        vec[i] = {static_cast<double>(data[i].real), static_cast<double>(data[i].imag)};
    }
    return vec;
}

/// @brief Reference implementation of the unnormalised DFT on the CPU, X_k = sum_n x_n * exp(-2 pi i k n / N)
std::vector<std::complex<double>> dft_cpu(const std::vector<bcomplex32> &input, int64_t const n_dfts, int64_t const dft_size)
{
    std::vector<std::complex<double>> output(input.size());
    for (int64_t row = 0; row < n_dfts; ++row)
    {
        for (int64_t k = 0; k < dft_size; ++k)
        {
            std::complex<double> sum = 0.0;
            for (int64_t n = 0; n < dft_size; ++n)
            {
                auto const &x = input[row * dft_size + n];
                double const angle = -2.0 * std::numbers::pi * static_cast<double>(k * n) / static_cast<double>(dft_size);
                sum += std::complex<double>(static_cast<double>(x.real), static_cast<double>(x.imag)) * std::polar(1.0, angle);
            }
            output[row * dft_size + k] = sum;
        }
    }
    return output;
}

/// @brief Finds the first available NPU that supports the DFT (currently only gen2p NPUs do).
/// @return The NPU id, or -1 if no such NPU is available.
int32_t find_dft_npu()
{
    size_t const n_npus_max = 20;
    uint32_t idxs[n_npus_max];
    char serials[n_npus_max][QANT_NATIVE_COMPUTING_TOOLKIT_DEFAULT_CHAR_ARR_LENGTH];
    size_t n_npus = 0;
    generic::QantDeviceCapability const operations[] = {generic::Dft};
    if (generic::get_available_npus(operations, 1, idxs, serials, n_npus_max, &n_npus) != 0 || n_npus == 0)
    {
        return -1;
    }
    // The matching NPUs are sorted by id, so this is the lowest id that supports the DFT.
    return static_cast<int32_t>(idxs[0]);
}

int main()
{
    int32_t const npu_id = find_dft_npu();
    if (npu_id < 0)
    {
        std::cerr << "no NPU supporting the DFT found" << std::endl;
        exit(-1);
    }
    std::cout << "running the DFT on NPU " << npu_id << std::endl;

    // each row is a signal made of two tones: a cosine with 5 + row cycles and a sine with 20 cycles
    int64_t const n_dfts = 4;
    int64_t const dft_size = 256;
    std::vector<bcomplex32> signal(n_dfts * dft_size);
    for (int64_t row = 0; row < n_dfts; ++row)
    {
        for (int64_t n = 0; n < dft_size; ++n)
        {
            double const t = static_cast<double>(n) / static_cast<double>(dft_size);
            double const value = std::cos(2.0 * std::numbers::pi * static_cast<double>(5 + row) * t) + 0.5 * std::sin(2.0 * std::numbers::pi * 20.0 * t);
            signal[row * dft_size + n] = {static_cast<std::bfloat16_t>(value), 0.0bf16};
        }
    }
    auto signal_tensor = vector_to_bcomplex_mtensor(signal, n_dfts, dft_size);

    auto res_npu_mtensor = native::dft(npu_id, &signal_tensor);
    if (res_npu_mtensor == nullptr)
    {
        exit(-1);
    }

    auto res_npu_vec = bcomplex_mtensor_to_vector(res_npu_mtensor);
    auto res_cpu_vec = dft_cpu(signal, n_dfts, dft_size);

    // compare the results with the relative L2 error of each row, as small frequency bins make the element-wise relative error meaningless
    double const error_threshold = 2e-2;
    bool errors_okay = true;
    for (int64_t row = 0; row < n_dfts; ++row)
    {
        double error_norm = 0.0;
        double reference_norm = 0.0;
        for (int64_t k = 0; k < dft_size; ++k)
        {
            error_norm += std::norm(res_npu_vec[row * dft_size + k] - res_cpu_vec[row * dft_size + k]);
            reference_norm += std::norm(res_cpu_vec[row * dft_size + k]);
        }
        double const relative_error = std::sqrt(error_norm / reference_norm);
        errors_okay = errors_okay && relative_error < error_threshold;

        // for a real signal the second half of the spectrum mirrors the first, so only the first half is searched for peaks
        std::cout << "DFT " << row << ": relative L2 error " << relative_error << ", peaks (frequency: npu / cpu magnitude):";
        for (int64_t k = 0; k < dft_size / 2; ++k)
        {
            double const magnitude_cpu = std::abs(res_cpu_vec[row * dft_size + k]);
            if (magnitude_cpu > static_cast<double>(dft_size) / 8.0)
            {
                std::cout << " " << k << ": " << std::abs(res_npu_vec[row * dft_size + k]) << " / " << magnitude_cpu;
            }
        }
        std::cout << std::endl;
    }

    generic::release_npu(npu_id);
    signal_tensor.deleter(&signal_tensor);
    res_npu_mtensor->deleter(res_npu_mtensor);

    if (!errors_okay)
    {
        std::cerr << "the NPU result differs too much from the CPU reference" << std::endl;
        return 1;
    }
    return 0;
}
