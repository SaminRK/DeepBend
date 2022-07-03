from numpy.testing._private.utils import assert_almost_equal
from pathlib import Path

import keras
import numpy as np

from models.prediction import Prediction, C0_PREDICT
from util.constants import RL, CNL
from util.reader import DNASequenceReader
from util.util import PathObtain


# TODO : Unit tests should finish fast


class TestPrediction:
    def test_load_model(self):
        pred = Prediction()
        assert isinstance(pred._model, keras.Model)

    def test_predict(self):
        df = DNASequenceReader().get_processed_data()[CNL].iloc[:100]
        result_df = Prediction().predict(df)
        assert C0_PREDICT in set(result_df.columns)

    def test_predict_model6(self):
        df = DNASequenceReader().get_processed_data()[CNL].iloc[:10]
        result_df = Prediction(6).predict(df)

        assert_almost_equal(
            np.round(result_df[C0_PREDICT], 3).tolist(),
            [0.122, -0.274, 0.606, 0.355, 0.106, -0.411, -0.993, -0.728, -0.461, 0.295],
            decimal=3,
        )

    def test_predict_model30(self):
        df = DNASequenceReader().get_processed_data()[CNL].iloc[:10]
        result_df = Prediction(30).predict(df)

        assert_almost_equal(
            np.round(result_df[C0_PREDICT], 3).tolist(),
            [
                -0.013,
                -0.465,
                0.531,
                0.204,
                0.241,
                -0.465,
                -1.284,
                -0.819,
                -0.314,
                0.283,
            ],
            decimal=3,
        )

    def test_predict_model35(self):
        df = DNASequenceReader().get_processed_data()[CNL].iloc[:10]
        result_df = Prediction(35).predict(df)

        assert_almost_equal(
            np.round(result_df[C0_PREDICT], 3).tolist(),
            [
                -0.080,
                -0.239,
                0.531,
                0.443,
                0.066,
                -0.498,
                -1.154,
                -0.448,
                -0.256,
                0.261,
            ],
            decimal=3,
        )

    def test_predict_lib(self):
        df = Prediction().predict_lib(RL, 100)
        assert len(df) == 101

    def test_predict_metrics_lib(self):
        Prediction().predict_metrics_lib(RL, 100)
        assert True

    def test_predict_metrics_lib_m30(self):
        Prediction(model=30).predict_metrics_lib(RL, 100)
