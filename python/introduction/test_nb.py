import os

import numpy as np
import testbook


def test_notebook():
    # Set working directory before testing for testing a collection of pytests
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    with testbook.testbook("./intro.ipynb", execute=True) as tb:
        error_dict_for_test = tb.get("error_dict_for_test")

        function_names = [
            "mul_elementwise",
            "calc_scaled_periodic_nl_fprop",
            "linear_fprop",
            "conv_fprop",
            "conv_transpose_fprop",
            "add_bias_fprop",
            "relu_fprop",
            "sigmoid_fprop",
            "softmax_fprop",
            "calc_kan_layer_fprop",
            "maxpool2d_fprop",
            "avgpool2d_fprop",
            "adaptive_maxpool2d_fprop",
            "adaptive_avgpool2d_fprop",
            "batchnorm2d_fprop",
        ]

        # for some functions, the error can be higher than 1e-2 due to numerical issues, but we want to make sure that most of them are below this threshold
        default_threshold = 5e-3
        special_errors = {
            "conv_transpose_fprop": 2e-2,
            "calc_scaled_periodic_nl_fprop": 2e-2,
            "softmax_fprop": 2e-2,
            "calc_kan_layer_fprop": 2e-1,  # very high error as we are testing the almost cosine against the exact cosine
            "batchnorm2d_fprop": 2e-2,
        }

        errors_okay = True

        for func_name in function_names:
            assert func_name in error_dict_for_test, (
                f"{func_name} is missing in error_dict_for_test"
            )
            error_threshold = special_errors.get(func_name, default_threshold)

            if np.mean(error_dict_for_test[func_name]) > error_threshold:
                errors_okay = False
                print(
                    f"{func_name} error is too high: {error_dict_for_test[func_name]}"
                )

        assert errors_okay, "Some functions have errors above the threshold"
