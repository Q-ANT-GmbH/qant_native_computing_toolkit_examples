import matrix_vector_multiplication as mvm
import numpy as np


def test():
    results_singleproc, results_multiproc = mvm.main()
    for res_single, res_multi in zip(results_singleproc, results_multiproc):
        np.testing.assert_allclose(res_single, res_multi, atol=1e-3)
