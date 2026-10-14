# Parallel Matrix-Vector Multiplication on Multiple NPUs
This project demonstrates how to parallelize a matrix-vector multiplication workload across multiple NPUs using Python's `multiprocessing`.

## Overview
Script: matrix_vector_multiplication.py
A batch of matrices is multiplied with a vector, once sequentially on a single NPU and once distributed across all NPUs reported by `qant.generic.get_available_npus()`. Matrices and the vector are shared between worker processes via `multiprocessing.shared_memory` to avoid pickling overhead, and each worker process is mapped to one NPU device.

## Dependencies
All required Python packages are listed in requirements.txt.

```bash
pip install -r requirements.txt 
```

## Usage
Run the script:
```bash
python matrix_vector_multiplication.py
```

## Testing
PyTest test case:
The file test_matrix_vector_parallel.py runs the single-NPU and multi-NPU code paths and asserts that their results match.

```bash
pytest test_matrix_vector_parallel.py 
```

# Description
Random matrices and a random vector are generated, then multiplied together once on a single NPU and once distributed across all available NPUs. The results of both approaches are compared, and the execution time of each approach is printed.
