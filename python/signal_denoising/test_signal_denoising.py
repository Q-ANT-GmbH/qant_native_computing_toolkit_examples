import os

import testbook


def test_notebook():
    # Set working directory before testing for testing a collection of pytests
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    with testbook.testbook("./signal_denoising_on_npu.ipynb", execute=True) as tb:
        mse_noisy = tb.get("mse_noisy")
        mse_denoised = tb.get("mse_denoised")

        # the denoised signal must be close to the clean sine
        assert mse_denoised < 0.1, f"denoised MSE too high: {mse_denoised}"
        # and much closer than the noisy input
        assert mse_denoised < 0.1 * mse_noisy, (
            f"denoising not effective: {mse_denoised=} vs. {mse_noisy=}"
        )
