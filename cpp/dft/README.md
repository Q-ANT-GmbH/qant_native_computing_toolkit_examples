# Discrete Fourier Transform (NPU vs CPU Implementation)
This project demonstrates a hardware-accelerated, batched discrete Fourier transform (DFT) on an NPU, with comparison to a reference DFT computed on the CPU.
It shows how complex numbers are passed to the NPU as `bcomplex32` DLPack tensors.
We use minimal dependencies, all you will need is a build system and DLPack.

## Overview
- Each row of a `(n_dfts, dft_size)` tensor is transformed independently with the unnormalised forward DFT $X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i k n / N}$.
- The input and output data type is `bcomplex32` (`kDLBcomplex`, 32 bits, 1 lane): each complex number is stored as two interleaved bfloat16 values (real, imaginary).
- The example finds an NPU that supports the DFT by passing the `Dft` capability to `get_available_npus`. Currently only gen2p NPUs support the DFT.
- The NPU result is compared to the CPU reference with the relative L2 error of each row. The example fails if the error exceeds `2e-2`.

## Prerequisites
Q.ANT Native Computing Toolkit (NPU driver/API)

C++23-compatible compiler (GCC/Clang)

CMake >= 3.16

DLPack (bundled as the `dlpack/` git submodule, pinned at v1.4 — run `git submodule update --init dlpack` if the directory is empty). `kDLBcomplex` is not available in older DLPack versions.

## Usage
Execute:
```bash
bash run_example.sh
```
