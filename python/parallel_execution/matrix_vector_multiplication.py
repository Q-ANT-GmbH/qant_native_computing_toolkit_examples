import concurrent.futures
import os
import time
from multiprocessing import shared_memory

import ml_dtypes
import numpy as np

import qant_native_computing_toolkit as qant


def multiply_matrix_vector_single(
    args: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Worker for the single-process approach
    """
    (matrix, vector) = args
    print(f"Worker[0 with pid {os.getpid()}] solves a job")
    return qant.native.linear_fprop(vector, matrix, device_id=0)


def multiply_matrix_vector_multi(
    args: tuple[str, str, int, int, int, int, dict[int, int]],
) -> np.ndarray:
    """
    Worker for the multi-process approach
    SharedMemory is used to share the matrix and the vector
    otherwise the pickle-overhead is the bottleneck
    """
    (
        matrices_shm_name,
        vector_shm_name,
        n_matrices,
        n_columns,
        n_rows,
        matrix_id,
        pid_to_wid,
    ) = args
    matrices_shm = shared_memory.SharedMemory(matrices_shm_name)
    matrices = np.ndarray(
        (n_matrices, n_rows, n_columns),
        dtype=ml_dtypes.bfloat16,
        buffer=matrices_shm.buf,
    )
    vector_shm = shared_memory.SharedMemory(vector_shm_name)
    vector = np.ndarray((n_rows), dtype=ml_dtypes.bfloat16, buffer=vector_shm.buf)
    # we are using the process id (pid) to identify the worker
    # each worker is linked to one NPU
    pid = os.getpid()
    wid = pid_to_wid[pid]
    print(f"Worker[{wid}] with pid {pid} solves a job")

    return qant.native.linear_fprop(vector, matrices[matrix_id], device_id=wid)


def getpid(_):
    """
    Helper function to collect the pid of all workers
    """
    # add a short delay, otherwise one process is maybe too fast and not all workers receive a pid entry
    time.sleep(0.001)
    return os.getpid()


def main():
    """
    In this example, we show a basic example on how to parallelise a workload for multiple NPUs.
    Given a batch of matrices to be multiplied with a vector, we use multiprocessing and SharedMemory
    to distribute parts of the batch to different NPUs.
    """
    n_matrices = 10
    n_columns = 2000
    n_rows = 2000

    rng = np.random.default_rng(42)
    matrices = (rng.uniform(-0.2, 0.2, size=(n_matrices, n_rows, n_columns))).astype(
        ml_dtypes.bfloat16
    )
    vector = (rng.uniform(-0.2, 0.2, size=(n_rows))).astype(ml_dtypes.bfloat16)

    npu_ids = list(qant.generic.get_available_npus().keys())
    n_npus = len(npu_ids)
    print(f"Found {n_npus} NPUs")
    task_args = [(matrix, vector) for matrix in matrices]

    time_start_singleproc = time.time()
    qant.generic.init_npu(0)
    results_singleproc = [multiply_matrix_vector_single(task) for task in task_args]
    qant.generic.release_npu(0)
    time_end_singleproc = time.time()

    print("--------------------")
    try:
        # We are using shared memory to avoid the pickling bottleneck
        matrices_shm = shared_memory.SharedMemory(create=True, size=matrices.nbytes)
        matrices_shm_arr = np.ndarray(
            matrices.shape, dtype=matrices.dtype, buffer=matrices_shm.buf
        )
        matrices_shm_arr[:] = matrices  # copy once

        vector_shm = shared_memory.SharedMemory(create=True, size=vector.nbytes)
        vector_shm_arr = np.ndarray(
            vector.shape, dtype=vector.dtype, buffer=vector_shm.buf
        )
        vector_shm_arr[:] = vector  # copy once

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_npus) as pool:
            # generate the pid to worker_id mapping, which is used to identify the NPU
            pids = set(pool.map(getpid, range(n_npus)))
            pids = sorted(pids)
            print(f"worker pids: {pids}")
            pid_to_wid = {pid: wid for wid, pid in zip(npu_ids, pids)}
            task_args_with_mapping = [
                (
                    matrices_shm.name,
                    vector_shm.name,
                    n_matrices,
                    n_columns,
                    n_rows,
                    matrix_id,
                    pid_to_wid,
                )
                for matrix_id in range(n_matrices)
            ]

            time_start_multiproc = time.time()
            results_multiproc = list(
                pool.map(multiply_matrix_vector_multi, task_args_with_mapping)
            )
            time_end_multiproc = time.time()
    finally:
        matrices_shm.close()
        matrices_shm.unlink()
        vector_shm.close()
        vector_shm.unlink()

    print(
        f"execution time single NPU {time_end_singleproc - time_start_singleproc:.1f}"
    )
    print(f"execution time multi NPU {time_end_multiproc - time_start_multiproc:.1f}")

    return results_singleproc, results_multiproc


if __name__ == "__main__":
    main()
