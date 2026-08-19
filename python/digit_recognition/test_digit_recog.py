import qant_native_computing_toolkit.ai as q_ai
import testbook
import numpy as np
import os

import utils


def qant_model(img, weights_layer_1, bias_layer_1, weights_layer_2, bias_layer_2):
    fm = q_ai.linear_fprop(img.flatten(), weights_layer_1)
    fm = q_ai.add_bias_fprop(fm, bias_layer_1)
    fm = q_ai.relu_fprop(fm)

    fm = q_ai.linear_fprop(fm, weights_layer_2)
    fm = q_ai.add_bias_fprop(fm, bias_layer_2)
    res = q_ai.softmax_fprop(fm).squeeze()

    return res


def test_digit_recog_accuracy():
    img, _, label = utils.load_random_image()
    weights_layer_1, bias_layer_1, weights_layer_2, bias_layer_2 = (
        utils.load_NN_weights()
    )

    pred = []
    expected = []

    for _ in range(1000):
        res = qant_model(
            img, weights_layer_1, bias_layer_1, weights_layer_2, bias_layer_2
        )
        digit = np.argmax(res)

        expected.append(label)
        pred.append(digit)

        img, _, label = utils.load_random_image()

    expected = np.array(expected)
    pred = np.array(pred)

    accuracy = (expected == pred).astype(float).mean()
    np.testing.assert_(accuracy > 0.85)


def test_notebook():
    # Set working directory before testing for testing a collection of pytests
    os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__))))

    with testbook.testbook("./digit_recognition_on_npu.ipynb", execute=True) as tb:
        label = tb.get("label")
        # convert np.int64 to normal int for comparison
        pred = int(tb.get("prediction").item())
        np.testing.assert_equal(pred, label)
