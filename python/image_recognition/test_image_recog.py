import os

import numpy as np
import testbook


def test_notebook():
    # Set working directory before testing for testing a collection of pytests
    os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__))))
    with testbook.testbook(
        "./image_recognition_on_npu.ipynb", execute=False, timeout=300
    ) as tb:
        # run every cell except the last (the benchmarking cell) so we can
        # override num_runs before it executes, keeping the test fast
        last_cell = len(tb.cells) - 1
        tb.execute_cell(range(last_cell))

        pred = tb.get("top5_label")[0]
        cert = tb.get("top5_prob_list")[0]
        np.testing.assert_equal(pred, "Maltese dog")
        np.testing.assert_array_less(0.8, cert)

        tb.inject("num_runs = 2", before=last_cell, run=True)
        tb.execute_cell(last_cell)
