# Matrix-Vector Multiplication (NPU vs CPU Implementation)
This project demonstrates hardware-accelerated matrix-vector multiplication using an NPU, with comparison to standard CPU execution. Ideal for understanding hardware acceleration benefits. 

## Prerequisites
Q.ANT Native Computing Toolkit (NPU driver/API)

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
The provided code multiplies a random matrix and vector.
