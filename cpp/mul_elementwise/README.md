# Vector-Vector elementwise multiplication (NPU vs CPU Implementation)
This project demonstrates hardware-accelerated vector-vector elementwise multiplication using an NPU, with comparison to standard CPU execution. 
We use minimal dependencies, all you will need is a build system and DLPack.

## Prerequisites
Q.ANT Native Computing Toolkit (NPU driver/API)

C++23-compatible compiler (GCC/Clang)

CMake >= 3.16

DLPack (bundled as the `dlpack/` git submodule, pinned at v1.2 — run `git submodule update --init dlpack` if the directory is empty)

## Usage
Execute:
```bash
bash run_examples.sh
```
