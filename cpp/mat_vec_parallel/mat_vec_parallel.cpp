/**
 * This file is an executable to demonstrate the basic usage of Q.ANT native computing toolkit
 * and how to parallelise a workload over multiple NPUs.
 * The example task is to compute many matrix vector multiplications by sending batches to worker threads.
 * We use xtensor as the cpp tensor backend. This is not a requirement.
 * Any tensor/data storage framework (eigen3, std::vector, ...) can be used,
 * as long as it can be converted to a DLPack tensor.
 */

#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdfloat> // for bfloat16_t (C++23)
#include <thread>
#include <vector>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "dlpack_utils.h"
#include "qant_native_computing_toolkit.h"

using namespace qant_native_computing_toolkit;
typedef xt::xarray<std::bfloat16_t> Tensor;

// wraps Q.ANT's linear_fprop by converting xarrays to/from DL tensors.
Tensor multiply_matrix_vector(
    const Tensor &matrix,
    const Tensor &vector,
    uint32_t device_id)
{
    auto vec_mtensor = xarray_to_mtensor_view(vector);
    auto mat_mtensor = xarray_to_mtensor_view(matrix);

    auto res_mtensor = native::linear_fprop(device_id, &vec_mtensor, &mat_mtensor);

    // release the DLPack views created above. This frees only the shape
    // arrays allocated by xarray_to_mtensor_view; the tensor data is still
    // owned by the matrix/vector arguments.
    vec_mtensor.deleter(&vec_mtensor);
    mat_mtensor.deleter(&mat_mtensor);

    if (res_mtensor == nullptr)
    {
        throw std::runtime_error("multiplication failed");
    }
    return mtensor_to_tensor(res_mtensor, true);
}

// multiplies a batch of matrices given by start and end index with the vector
// and places the result in results.
void worker(
    const std::vector<Tensor> &matrices,
    const Tensor &vector,
    std::vector<Tensor> &results,
    size_t start,
    size_t end,
    uint32_t device_id)
{
    for (size_t i = start; i < end; ++i)
    {
        results[i] = multiply_matrix_vector(matrices[i], vector, device_id);
    }
}

int main()
{
    auto n_matrices = 100;
    auto n_columns = 1000;
    auto n_rows = 1000;

    std::cout << "setting up test data" << std::endl;
    // create random matrices and vector
    auto matrices = std::vector<Tensor>();
    matrices.reserve(n_matrices);
    xt::xarray<float> matrix_float = xt::random::rand({n_rows, n_columns}, -0.1, 0.1);
    Tensor matrix = xt::cast<std::bfloat16_t>(matrix_float);
    for (size_t i = 0; i < n_matrices; ++i)
    {
        matrices.push_back(matrix);
    }
    xt::xarray<float> vector_float = xt::random::rand({1, n_rows}, -0.1, 0.1);
    Tensor vector = xt::cast<std::bfloat16_t>(vector_float);

    // find out which NPUs are available on the system
    size_t const n_npus_max = 10;
    uint32_t device_ids[n_npus_max];
    char serials[n_npus_max][QANT_NATIVE_COMPUTING_TOOLKIT_DEFAULT_CHAR_ARR_LENGTH];
    size_t n_npus = 0;

    int const err = info::get_available_npus(device_ids, serials, n_npus_max, &n_npus);
    if (err != 0)
    {
        throw std::runtime_error("get_available_npus() failed");
    };
    if (n_npus == 0)
    {
        throw std::runtime_error("no NPUs found on this system");
    }

    // perform dummy operation such that performance measurement later is not affected
    // by initialisation and calibration.
    std::cout << "warming up npus [";
    for (size_t i = 0; i < n_npus; i++)
        std::cout << (i ? ", " : "") << device_ids[i];
    std::cout << "]" << std::endl;
    for (int i = 0; i < n_npus; i++)
    {
        std::cout << "npu id " << device_ids[i] << " ..." << std::endl;
        multiply_matrix_vector(matrices[0], vector, device_ids[i]);
    }

    // create a thread for each NPU and send batches of matrices to be worked on
    std::vector<Tensor> results_multi(n_matrices);
    std::vector<Tensor> results_single(n_matrices);

    std::vector<std::thread> threads;
    threads.reserve(n_npus);

    auto chunk_size = n_matrices / n_npus;
    auto remainder = n_matrices % n_npus;

    auto start_idx = 0;
    std::cout << "starting calculation" << std::endl;
    auto start_multi = std::chrono::high_resolution_clock::now();
    for (size_t t = 0; t < n_npus; ++t)
    {
        auto end_idx = start_idx + chunk_size + (t < remainder ? 1 : 0);

        threads.emplace_back(
            worker,
            std::cref(matrices),
            std::cref(vector),
            std::ref(results_multi),
            start_idx,
            end_idx,
            device_ids[t]);

        start_idx = end_idx;
    }

    for (auto &th : threads)
        th.join();
    auto end_multi = std::chrono::high_resolution_clock::now();

    // for comparison: execute multiplications sequential
    auto start_single = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n_matrices; ++i)
    {
        uint device_id = i % n_npus;
        results_single[i] =
            multiply_matrix_vector(matrices[i], vector, device_id);
    }
    auto end_single = std::chrono::high_resolution_clock::now();

    auto time_multi =
        std::chrono::duration<double>(end_multi - start_multi).count();

    auto time_single =
        std::chrono::duration<double>(end_single - start_single).count();

    std::cout << "execution time singlethreaded: "
              << time_single << " s"
              << std::endl;

    std::cout << "execution time multithreaded: "
              << time_multi << " s"
              << std::endl;

    // Optional step: release the NPUs to free the memory held by the driver.
    for (size_t i = 0; i < n_npus; ++i)
    {
        generic::release_npu(device_ids[i]);
    }

    return 0;
}
