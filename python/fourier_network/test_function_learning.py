import testbook
import numpy as np
import os


def test_notebook():
    # Set working directory before testing for testing a collection of pytests
    os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__))))
    with testbook.testbook(
        "./function_learning_on_npu.ipynb", execute=True, timeout=300
    ) as tb:
        mlp_train_losses = tb.get("mlp_train_losses")
        fourier_train_losses = tb.get("fourier_train_losses")
        np.testing.assert_array_less(mlp_train_losses[-1], 0.05)
        np.testing.assert_array_less(fourier_train_losses[-1], 0.005)
