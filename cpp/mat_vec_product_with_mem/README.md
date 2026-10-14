# Matrix-Vector Multiplication with explicit memory management (NPU vs CPU Implementation)
This project demonstrates hardware-accelerated matrix-vector multiplication using an NPU via the
toolkit's explicit memory-management API (`generic::npu_upload`/`npu_allocate`/`npu_upload_into`/
`native::linear_fprop_preloaded`/`npu_download`/`npu_free`), with comparison to standard CPU execution. Unlike [../mat_vec_product](../mat_vec_product),
which performs the whole operation with a single `native::linear_fprop()` call, this example allocates and
uploads the weights and features to NPU memory itself and executes the product directly on that resident
data. This is useful when the same matrix (e.g. a set of trained weights) should stay resident on the
NPU across many calls, since only the (much smaller) features/result then need to be transferred per call.

## Prerequisites
Q.ANT Native Computing Toolkit (NPU driver/API), built with the memory-management feature

C++23-compatible compiler (GCC/Clang)

CMake >= 3.16

XTensor == 0.25.0

XTensor-BLAS == 0.21.0

DLPack (bundled as the `dlpack/` git submodule, pinned at v1.2 — run `git submodule update --init dlpack` if the directory is empty)

## Usage
Execute:
```bash
bash run_example.sh
```

## Description
The provided code uploads random weights to NPU memory, allocates NPU memory for random features
and the result, uploads the features into their allocated memory, executes the linear forward
propagation (`features @ weights^T`) on the NPU, downloads the result, and compares it to the same product computed
on the CPU.
